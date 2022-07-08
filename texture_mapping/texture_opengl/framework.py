from OpenGL.GL.framebufferobjects import glBindFramebuffer
from texture_mapping.texture_opengl.arcball.ArcBall import *
from OpenGL.GLUT import *
from OpenGL.GL import *
import numpy as np
import cv2
from abc import ABC, abstractmethod

class OpenglRendererBase():
    glutWindowShared = None

    def __init__(self, name):
        self.frameBufferObject = None
        self.frameBufferRenderBufferObject = None
        self.frameBufferTextureObject = None
        self.name = name
        self.window_name = 'rendered'
        self.drawers = []

        self.frame_counter = 0
        self.record_path_format = None
        self.record_video_writer = None
        self.record_speed = None


    def init(self, window_name, H, W, drawers):
        self.window_name = window_name
        self.H = H
        self.W = W

        self.drawers = drawers

        OpenglRendererBase.initializeGlut()
        glutSetWindowTitle(self.window_name)

    def display(self, camera_matrix=None):
        glViewport(0, 0, self.W, self.H)
        glClearColor(1.0, 1.0, 1.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_DEPTH_TEST)
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

        for drawer in self.drawers:
            drawer.draw(camera_matrix, frameBufferObject=self.frameBufferObject)

        glutSwapBuffers()
        glutPostRedisplay()

        if self.record_path_format is not None or self.record_video_writer is not None:
            if self.frame_counter % self.record_speed == 0:
                frame = ((self.screenshot()[:,:,::-1]).clip(0, 1) * 255).astype(np.uint8)
                if self.record_path_format is not None:
                    image_path = self.record_path_format % self.frame_counter
                    print(f'[OpenglRendererBase] Save frame: {image_path}')
                    cv2.imwrite(image_path, frame)
                if self.record_video_writer is not None:
                    self.record_video_writer.write(frame)
            self.frame_counter += 1

    def screenshot(self, displaying=False, pause=100):
        image = glReadPixels(0, 0,
                             self.W,
                             self.H,
                             GL_RGB,
                             GL_FLOAT)
        image = np.frombuffer(image, dtype='float32').reshape((self.H, self.W, 3))
        image = image[::-1, :, :]

        if displaying:
            cv2.imshow(self.window_name, image)
            keypress = cv2.waitKey(pause)
            return image, keypress

        return image

    def initializeFrameBuffer(self):
        if True or self.frameBufferObject is None:
            self.frameBufferObject = glGenFramebuffers(1)

        if True or self.frameBufferTextureObject is None:
            self.frameBufferTextureObject = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.frameBufferTextureObject)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, self.W, self.H, 0, GL_RGB, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        if True or self.frameBufferRenderBufferObject is None:
            self.frameBufferRenderBufferObject = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.frameBufferRenderBufferObject)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self.W, self.H)

        glBindFramebuffer(GL_FRAMEBUFFER, self.frameBufferObject)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.frameBufferTextureObject, 0)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.frameBufferRenderBufferObject)
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print('Framebuffer not complete')
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    @staticmethod
    def initializeGlut():
        if OpenglRendererBase.glutWindowShared is not None:
            return
        glutInit()
        displayMode = GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH | GLUT_STENCIL;
        glutInitDisplayMode(displayMode)
        glutInitWindowSize(100, 100)
        glutInitWindowPosition(100, 100)
        OpenglRendererBase.glutWindowShared = glutCreateWindow('glutWindowShared')

    @staticmethod
    def loadShaderFromFile(shaderType, shaderFile):
        strFilename = OpenglRendererBase.findFileOrThrow(shaderFile)
        with open(strFilename, 'r') as f:
            shaderData = f.read()
        return OpenglRendererBase.loadShader(shaderType, shaderData)

    @staticmethod
    def loadShader(shaderType, shaderData):
        shader = glCreateShader(shaderType)
        glShaderSource(shader, shaderData)

        glCompileShader(shader)

        status = glGetShaderiv(shader, GL_COMPILE_STATUS)
        if status == GL_FALSE:
            strInfoLog = glGetShaderInfoLog(shader)
            strShaderType = ""
            if shaderType is GL_VERTEX_SHADER:
                strShaderType = "vertex"
            elif shaderType is GL_GEOMETRY_SHADER:
                strShaderType = "geometry"
            elif shaderType is GL_FRAGMENT_SHADER:
                strShaderType = "fragment"

            print("Compilation failure for ", strShaderType, " data:\n", strInfoLog)

        return shader

    @staticmethod
    def createProgram(shaderList):
        program = glCreateProgram()

        for shader in shaderList:
            glAttachShader(program, shader)

        glLinkProgram(program)

        status = glGetProgramiv(program, GL_LINK_STATUS)
        if status == GL_FALSE:
            strInfoLog = glGetProgramInfoLog(program)
            print("Linker failure: \n", strInfoLog)

        for shader in shaderList:
            glDetachShader(program, shader)

        return program

    @staticmethod
    def findFileOrThrow(shaderFile):
        if os.path.isfile(shaderFile):
            return shaderFile
        raise IOError('Could not find target file ' + shaderFile)

class OpenglRendererPlanar(OpenglRendererBase):

    def render(self, H, W, nVertices, vertexDim, coordData, colorData, window_name='rendered'):

        #------------- initialize rendering background -------------#
        self.init(
            window_name, H, W,
            drawers=[
                OpenglDrawerTexturedMesh(
                    'planar_rasterize.vert',
                    'barycentric_colors.frag',
                    nVertices, vertexDim, coordData, colorData
                )
            ]
        )
        self.initializeFrameBuffer()

        self.display()

        return self.screenshot()

class OpenglRendererPerspectiveWorldCoord(OpenglRendererBase):

    def __init__(self, name, H, W, nVertices, vertexDim, coordData):
        super().__init__(name)

        #------------- initialize rendering background -------------#
        self.init(
            name, H, W,
            drawers=[
                OpenglDrawerTexturedMesh(
                    'perspective_depthmap.vert',
                    'barycentric_colors.frag',
                    nVertices, vertexDim, coordData, coordData
                )
            ],
        )

    def render(self, camera_matrix_list, extrinsics=np.eye(4)[None,...].repeat(6, axis=0)):
        self.initializeFrameBuffer()
        rendered = []
        extrinsicMatrixLocation = glGetUniformLocation(self.drawers[0].theProgram, "extrinsicMatrix")
        for i, camera_matrix in enumerate(camera_matrix_list):
            glUseProgram(self.drawers[0].theProgram)
            glUniformMatrix4fv(
                extrinsicMatrixLocation,
                1, GL_FALSE,
                extrinsics[i].T.reshape(-1).astype('float32'))
            glUseProgram(0)

            self.display(camera_matrix)
            rendered.append(self.screenshot())
        return rendered

class OpenglRendererArcball(OpenglRendererBase):

    def render(self, H, W, camera_matrix_looper, drawers, record_path_format=None, record_speed=1, record_one_cycle_only=1, initial_rotation=np.eye(3)):

        if record_path_format is not None:

            record_path_format = os.path.splitext(record_path_format)[0] + '.avi'
            self.record_video_writer = cv2.VideoWriter(
                record_path_format,
                cv2.VideoWriter_fourcc(*'XVID'),
                30.0,
                (W, H)
            )
            self.record_path_img_format = os.path.splitext(record_path_format)[0] + '_%04d.png'

            self.record_speed = record_speed
            self.record_one_cycle_only = record_one_cycle_only
            print(f'[OpenglRendererArcball.render] Recording a video at speed {record_speed}: {record_path_format}')

        print("initial_rotation", initial_rotation)
        self.initial_rotation = initial_rotation

        #------------- initialize rendering background -------------#
        self.init(
            self.name, H, W,
            drawers=drawers
        )

        self.__initializeArcball(W, H)

        glutReshapeWindow(W, H)

        #------------- calculate reference camera extrinsic -------------#
        self.camera_matrix_looper = camera_matrix_looper
        self.camera_matrix_looper.set_extrinsic_ref(self.extrinsic_ref[self.exti])
        self.pinned = False
        self.initial_hfov = round(self.camera_matrix_looper.get_hfov())
        self.hfov = self.initial_hfov

        #------------- begin glut loop -------------#
        glutDisplayFunc(self.display_acrball)
        glutIdleFunc(self.display_acrball)
        glutKeyboardFunc(self.glut_event_keyboard)
        glutMouseFunc(self.glut_event_mouse)
        glutMotionFunc(self.glut_event_drag)
        glutMainLoop()

    def display_acrball(self):
        camera_matrix = self.camera_matrix_looper.progress()
        self.display(camera_matrix=camera_matrix)

    def __initializeArcball(self, W, H):
        self.LastRot = [self.initial_rotation, self.initial_rotation]
        self.ThisRot = [self.initial_rotation, self.initial_rotation]
        self.ThisTrans = [Vector3fT(), Vector3fT()]
        self.exti = 0
        self.exti = 1
        self.extrinsic_ref = [self.get_current_extrinsic()] * 2

        self.ArcBall = ArcBallT(W, H)
        self.isDragging = False
        self.quadratic = None

    def glut_event_keyboard(self, key, *args):
        key = key[0]

        if key == 27:       #   esc
            sys.exit()
        elif key in [91, 93]:     #   '[',']'
            for drawer in self.drawers:
                if key == 91:
                    drawer.texture_move_prev()
                else:
                    drawer.texture_move_next()
            extrinsic = self.get_current_extrinsic()

            if (not self.pinned) and self.record_video_writer is not None and self.record_one_cycle_only and self.get_current_i() == 0:
                self.record_video_writer.release()
                glutLeaveMainLoop()

            self.exti = 1
            self.extrinsic_ref[self.exti] = extrinsic
            self.ThisTrans[self.exti] = Vector3fT()
            self.camera_matrix_looper.set_extrinsic_ref(
                self.extrinsic_ref[self.exti],
                translation_3=self.ThisTrans[self.exti],
                rotation_mat_3x3=self.ThisRot[self.exti]
            )
            self.update_window_title()
        elif key == ord('p'):   # pin
            self.pinned = ~self.pinned
            self.update_window_title()

            self.camera_matrix_looper.pin_translation_offset()
        elif key == ord('c'): # capture
            frame = ((self.screenshot()[:, :, ::-1]).clip(0, 1) * 255).astype(np.uint8)
            image_path = "capture.png"
            print(f'[OpenglRendererBase] Save frame: {image_path}')
            cv2.imwrite(image_path, frame)

        elif key in [ord('d'), ord('D')]:
            colorDisparity = 1 if key == ord('d') else 0
            for drawer in self.drawers:
                uniform_loc = glGetUniformLocation(drawer.theProgram, "colorDisparity")
                if uniform_loc != -1:
                    glUseProgram(drawer.theProgram)
                    glUniform1i(uniform_loc, colorDisparity)
                    glUseProgram(0)
        elif key == ord('v'):   # toggle extrinsic
            print("self extrinsic ref", self.extrinsic_ref)
            print("self This trans  ref", self.ThisTrans)
            print("self This rot ref", self.ThisRot)
            self.exti = 1 - self.exti
            print(f"view[{'center' if self.exti == 0 else 'current'}]")
            self.camera_matrix_looper.set_extrinsic_ref(
                self.extrinsic_ref[self.exti],
                translation_3=self.ThisTrans[self.exti],
                rotation_mat_3x3=self.ThisRot[self.exti]
            )
        elif key == ord(' '):   # spoid extrinsic
            self.exti = 1
            extrinsic = self.get_current_extrinsic()
            self.extrinsic_ref[self.exti] = extrinsic
            self.reset_extrinsic_ref()

        elif ord('0') <= key <= ord('9'):    # move up, down, left, right
            movement = np.array([0, 0, 0])
            d = 0.1
            if   key == ord('8'):     movement[1] = -1
            elif key == ord('2'):   movement[1] = +1
            elif key == ord('4'):   movement[0] = +1
            elif key == ord('6'):  movement[0] = -1
            elif key == ord('1'):  movement[2] = +1
            elif key == ord('3'):   movement[2] = -1
            self.ThisTrans[self.exti] = self.camera_matrix_looper.translate_circenter(d * movement)
        elif key == 53:   # reset rotation
            self.LastRot[self.exti] = self.initial_rotation
            self.ThisRot[self.exti] = self.initial_rotation
            self.camera_matrix_looper.rotate_circenter(self.ThisRot[self.exti])

    def reset_extrinsic_ref(self):
        self.LastRot[self.exti] = self.initial_rotation
        self.ThisRot[self.exti] = self.initial_rotation
        self.ThisTrans[self.exti] = Vector3fT()
        self.camera_matrix_looper.set_extrinsic_ref(self.extrinsic_ref[self.exti])

    def glut_event_drag(self, cursor_x, cursor_y):
        if (self.isDragging):
            mouse_pt = Point2fT(cursor_x, cursor_y)
            ThisQuat = self.ArcBall.drag(mouse_pt)
            self.ThisRot[self.exti] = Matrix3fSetRotationFromQuat4f(ThisQuat)
            self.ThisRot[self.exti] = Matrix3fMulMatrix3f(self.ThisRot[self.exti], self.LastRot[self.exti])
            self.camera_matrix_looper.rotate_circenter(self.ThisRot[self.exti])

    def glut_event_mouse(self, button, button_state, cursor_x, cursor_y):
        self.isDragging = False
        if (button == GLUT_RIGHT_BUTTON and button_state == GLUT_UP):
            # Right button click
            self.reset_extrinsic_ref()
            extrinsic = self.get_current_extrinsic()
            self.extrinsic_ref[self.exti] = extrinsic
            self.camera_matrix_looper.set_extrinsic_ref(extrinsic)
        elif (button == GLUT_LEFT_BUTTON and button_state == GLUT_UP):
            # Left button released
            self.LastRot[self.exti] = copy.copy(self.ThisRot[self.exti])
        elif (button == GLUT_LEFT_BUTTON and button_state == GLUT_DOWN):
            # Left button clicked down
            self.LastRot[self.exti] = copy.copy(self.ThisRot[self.exti])
            self.isDragging = True
            mouse_pt = Point2fT(cursor_x, cursor_y)
            self.ArcBall.click(mouse_pt)
        elif (button == 3 or button == 4) and button_state == GLUT_DOWN:
            self.hfov = self.camera_matrix_looper.set_hfov(self.hfov + (-1 if button == 3 else +1))
            self.update_window_title()
        elif (button == GLUT_MIDDLE_BUTTON and button_state == GLUT_UP):
            # mouse wheel click
            self.hfov = self.camera_matrix_looper.set_hfov(self.initial_hfov)
            self.update_window_title()

    def get_current_extrinsic(self):
        extrinsic = None
        for drawer in self.drawers:
            if drawer.texture_map_provider is not None:
                extrinsic = drawer.texture_map_provider.get_current_extrinsic()
        return extrinsic

    def get_current_i(self):
        current_i = None
        for drawer in self.drawers:
            if drawer.texture_map_provider is not None:
                current_i = drawer.texture_map_provider.get_current_i()
        return current_i

    def update_window_title(self):
        glutSetWindowTitle(
            f'{self.drawers[-1].texture_map_provider.get_current_name()} (pinned={bool(self.pinned)}) (fov={self.hfov})'
        )


class OpenglDrawerBase(ABC):
    def __init__(self):
        self.theProgram = None
        self.vertexBufferObject = None
        self.vertexArrayObject = None

        self.textureObject = None

        self.texture_map_provider = None
        OpenglRendererBase.initializeGlut()

    @abstractmethod
    def draw(self, camera_matrix, frameBufferObject=None):
        pass

    def set_texture(self):
        pass

    def texture_move_next(self):
        if self.texture_map_provider is not None:
            texture, name = self.texture_map_provider.get_next()
            self.set_texture(texture)

    def texture_move_prev(self):
        if self.texture_map_provider is not None:
            texture, name = self.texture_map_provider.get_prev()
            self.set_texture(texture)
            return name


class OpenglDrawerTexturedMesh(OpenglDrawerBase):

    def __init__(self, vertex_shader_file, fragment_shader_file, nVertices, vertexDim, coordData, colorData, texture_map_provider=None):
        super().__init__()

        self.vertexDim = vertexDim
        self.nVertices = nVertices
        vertexData = self.serialize_vertex_data(coordData, colorData)

        shader_dir = 'texture_mapping/texture_opengl'
        self.__initializeProgram(
            os.path.join(shader_dir, vertex_shader_file),
            os.path.join(shader_dir, fragment_shader_file)
        )
        self.__initializeVertexBuffer(vertexData)

        self.vertexArrayObject = glGenVertexArrays(1)

        if texture_map_provider is not None:
            self.texture_map_provider = texture_map_provider
            self.set_texture(self.texture_map_provider.get_current()[0])

    def draw(self, camera_matrix, frameBufferObject=None):
        glUseProgram(self.theProgram)

        #####[ bind buffer ]#####################################
        glBindVertexArray(self.vertexArrayObject)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBufferObject)
        glEnableVertexAttribArray(0)    # vertex position
        glEnableVertexAttribArray(1)    # vertex color

        # bind buffer - vertex position
        glVertexAttribPointer(0, self.vertexDim, GL_FLOAT, GL_FALSE, 0, None)

        # bind buffer - vertex color
        colorOffset = c_void_p(self.vertexDim * self.nVertices * 4)
        glVertexAttribPointer(1, self.vertexDim, GL_FLOAT, GL_FALSE, 0, colorOffset)

        #####[ bind framebuffer ]#################################
        if frameBufferObject is not None:
            glBindFramebuffer(GL_FRAMEBUFFER, frameBufferObject)
        else:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)

        #####[ bind texture ]#################################
        if self.textureObject is not None:
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.textureObject)

        #####[ set camera_matrix ]#################################
        if camera_matrix is not None:
            glUniformMatrix4fv(
                glGetUniformLocation(self.theProgram, "perspectiveMatrix"),
                1, GL_FALSE,
                camera_matrix.T.reshape(-1).astype('float32'))

        #####[ actual drawing ]###################################
        glDrawArrays(GL_TRIANGLES, 0, self.nVertices)

        #####[ finalize ]#########################################
        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)
        glBindVertexArray(0)
        glUseProgram(0)


    def set_texture(self, texture_map):
        if texture_map is None:
            return

        if not isinstance(texture_map, np.ndarray):
            texture_map = texture_map.cpu().numpy()

        assert (texture_map.ndim == 2 or texture_map.ndim == 3)
        if texture_map.ndim == 2:
            texture_map = texture_map[..., None]

        assert (texture_map.shape[2] == 1 or texture_map.shape[2] == 3)
        if texture_map.shape[2] == 1:
            texture_map = texture_map.repeat(3, axis=2)

        if self.textureObject is None:
            self.textureObject = glGenTextures(1)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.textureObject)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_map.shape[1], texture_map.shape[0],
                     0, GL_RGB, GL_UNSIGNED_BYTE, texture_map[::-1].astype('uint8'))
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glUseProgram(self.theProgram)
        glUniform1i(
            glGetUniformLocation(self.theProgram, "ourTexture"),
            1
        )
        glUseProgram(0)


    def __initializeProgram(self, vertex_shader_path, fragment_shader_path):
        shaderList = []
        shaderList.append(OpenglRendererBase.loadShaderFromFile(GL_VERTEX_SHADER, vertex_shader_path))
        shaderList.append(OpenglRendererBase.loadShaderFromFile(GL_FRAGMENT_SHADER, fragment_shader_path))

        self.theProgram = OpenglRendererBase.createProgram(shaderList)

        for shader in shaderList:
            glDeleteShader(shader)

        glUseProgram(0)

    def __initializeVertexBuffer(self, vertexData):
        self.vertexBufferObject = glGenBuffers(1)

        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBufferObject)
        glBufferData(  # PyOpenGL allows for the omission of the size parameter
            GL_ARRAY_BUFFER,
            vertexData,
            GL_STREAM_DRAW
        )
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def serialize_vertex_data(self, coord_data, color_data):
        vertex_data = np.concatenate(
            [self.__pad_vertex_data(data).reshape(-1) for data in [coord_data, color_data]])
        return vertex_data.astype('float32')

    def __pad_vertex_data(self, data):
        if not isinstance(data, np.ndarray):
            data = data.cpu().numpy()
        # pad with 1 along the last dimension
        data = data.reshape((-1, data.shape[-1]))
        return np.hstack([
            data,
            np.ones((data.shape[0], self.vertexDim - data.shape[1]))
        ])


class OpenglDrawerCubemap(OpenglDrawerBase):
    def __init__(self, cubeface_images=None, texture_map_provider=None):
        super().__init__()

        self.cubemapTextureObject = None

        self.__set_cubemap_shaders()
        self.__initializeVertexBuffer()
        self.vertexArrayObject = glGenVertexArrays(1)

        if texture_map_provider is not None:
            self.texture_map_provider = texture_map_provider
            self.set_texture(self.texture_map_provider.get_current()[0])
        else:
            self.set_texture(cubeface_images)


    def draw(self, camera_matrix, frameBufferObject=None):
        glUseProgram(self.theProgram)

        #####[ bind buffer ]#####################################
        glBindVertexArray(self.vertexArrayObject)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBufferObject)
        glEnableVertexAttribArray(0)    # vertex position
        # glEnableVertexAttribArray(1)    # vertex color

        # bind buffer - vertex position
        glVertexAttribPointer(0, self.vertexDim, GL_FLOAT, GL_FALSE, 0, None)

        #####[ bind framebuffer ]#################################
        if frameBufferObject is not None:
            glBindFramebuffer(GL_FRAMEBUFFER, frameBufferObject)

        #####[ bind texture ]#################################
        glActiveTexture(GL_TEXTURE0)
        # glBindTexture(GL_TEXTURE_2D, self.textureObject)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.cubemapTextureObject)

        #####[ set camera_matrix ]#################################
        glUniformMatrix4fv(
            glGetUniformLocation(self.theProgram, "projectionMatrix"),
            1, GL_FALSE,
            np.hstack([camera_matrix[:,:3], np.array([0,0,1,0]).reshape((-1,1))]).T.reshape(-1).astype('float32'))

        #####[ actual drawing ]###################################
        # glDrawArrays(GL_TRIANGLES, 0, self.nVertices)
        glDrawArrays(GL_TRIANGLES, 0, self.nVertices);

        #####[ finalize ]#########################################
        glDisableVertexAttribArray(0)
        # glDisableVertexAttribArray(1)
        glBindVertexArray(0)
        glUseProgram(0)


    def set_texture(self, cubeface_images):
        if self.cubemapTextureObject is None:
            self.cubemapTextureObject = glGenTextures(1)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.cubemapTextureObject)

        H, W = cubeface_images[0].shape[:2]
        for i, cubeface_image in enumerate(cubeface_images):
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, W, H, 0, GL_RGB, GL_FLOAT, cubeface_image)

        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)


    ############################################################
    # private method
    ############################################################
    def __initializeVertexBuffer(self):
        self.vertexBufferObject = glGenBuffers(1)

        cubemap_vertices = self.__cubemap_vertices()
        self.vertexDim = 3
        self.nVertices = cubemap_vertices.size // 3

        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBufferObject)
        glBufferData(  # PyOpenGL allows for the omission of the size parameter
            GL_ARRAY_BUFFER,
            cubemap_vertices,
            GL_STREAM_DRAW
        )
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def __set_cubemap_shaders(self):
        v_shader_source = [
            """
            #version 330 core
            layout (location = 0) in vec3 aPos;
            
            out vec3 TexCoords;
            
            uniform mat4 projection;
            uniform mat4 view;
            uniform mat4 projectionMatrix;
            
            void main()
            {
                TexCoords = aPos;
                //vec4 pos = projection * view * vec4(aPos, 1.0);
                vec4 pos = projectionMatrix * vec4(aPos, 1.0);
                gl_Position = pos.xyww;
            }  
            """
        ]
        f_shader_source = [
            """
            #version 330 core
            out vec4 FragColor;
    
            in vec3 TexCoords;
    
            uniform samplerCube skybox;
    
            void main()
            {
                FragColor = texture(skybox, TexCoords);
            }
            """
        ]
        shaderList = []
        shaderList.append(OpenglRendererBase.loadShader(GL_VERTEX_SHADER, v_shader_source))
        shaderList.append(OpenglRendererBase.loadShader(GL_FRAGMENT_SHADER, f_shader_source))

        self.theProgram = OpenglRendererBase.createProgram(shaderList)

        for shader in shaderList:
            glDeleteShader(shader)

        glUseProgram(0)

    def __cubemap_vertices(self):
        return np.array([
            -1.0,  1.0, -1.0,
            -1.0, -1.0, -1.0,
             1.0, -1.0, -1.0,
             1.0, -1.0, -1.0,
             1.0,  1.0, -1.0,
            -1.0,  1.0, -1.0,

            -1.0, -1.0,  1.0,
            -1.0, -1.0, -1.0,
            -1.0,  1.0, -1.0,
            -1.0,  1.0, -1.0,
            -1.0,  1.0,  1.0,
            -1.0, -1.0,  1.0,

             1.0, -1.0, -1.0,
             1.0, -1.0,  1.0,
             1.0,  1.0,  1.0,
             1.0,  1.0,  1.0,
             1.0,  1.0, -1.0,
             1.0, -1.0, -1.0,

            -1.0, -1.0,  1.0,
            -1.0,  1.0,  1.0,
             1.0,  1.0,  1.0,
             1.0,  1.0,  1.0,
             1.0, -1.0,  1.0,
            -1.0, -1.0,  1.0,

            -1.0,  1.0, -1.0,
             1.0,  1.0, -1.0,
             1.0,  1.0,  1.0,
             1.0,  1.0,  1.0,
            -1.0,  1.0,  1.0,
            -1.0,  1.0, -1.0,

            -1.0, -1.0, -1.0,
            -1.0, -1.0,  1.0,
             1.0, -1.0, -1.0,
             1.0, -1.0, -1.0,
            -1.0, -1.0,  1.0,
             1.0, -1.0,  1.0
        ], dtype='float32')