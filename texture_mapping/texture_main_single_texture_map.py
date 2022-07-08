import functools
from stl.mesh import Mesh
from texture_mapping.equitriangle_tiler import *
from texture_mapping.viewer import *
from texture_mapping.equirect_depth_renderer import *
from sklearn.cluster import KMeans

def select_frames(translations, key_frame_num, frame_idx_list):
    model = KMeans(n_clusters=key_frame_num)
    translations_numpy = translations.cpu().numpy()
    model.fit(translations_numpy)
    means = model.cluster_centers_
    return_idx_list = []

    for i in range(len(means)):
        mean = means[i]
        cost = translations_numpy - mean
        idx = np.argmin(np.linalg.norm(cost, axis=1, keepdims=True))
        return_idx_list.append(frame_idx_list[idx])
    return return_idx_list

def main(args):
    timestamps = []

    tic()
    ## Load input videos and extriniscs
    input_video_dir = os.path.join(args.data_root_path, args.dir_input_video, args.video_name)
    extrinsics_csv_path = os.path.join(input_video_dir, args.input_trajectory_name + ".csv")
    total_video_frames_count, extrinsics = load_extrinsics_from_csv(extrinsics_csv_path, args.max_video_frames_count)
    center2world = calculate_center2world(extrinsics)
    extrinsics = torch.matmul(extrinsics, center2world)


    #------------- set paths -------------#
    texture_map_max_sizeK = args.texture_map_max_sizeK

    semicasename = f'{args.video_name}_{args.stl_name}_ST'
    casename = f'{semicasename}_tri{texture_map_max_sizeK}_key_{args.key_frame_num}'
    local_texture_output_path = os.path.join(args.data_root_path, args.dir_rendering_output,
                                             args.video_name,
                                             semicasename,
                                             casename)

    texture_output_path = local_texture_output_path
    output_path_prefix = os.path.join(texture_output_path, casename)

    #------------- frame selection -------------#
    frame_idx_list = range(0, total_video_frames_count, args.texture_frame_idx_list_step)
    translations = extrinsic2translation(extrinsics[frame_idx_list])  # CAM * XYZ
    frame_idx_list_final = select_frames(translations, args.key_frame_num, frame_idx_list)

    video = cv2.VideoCapture(os.path.join(input_video_dir, "video.mp4"))

    #------------- load mesh -------------#
    stl_path = os.path.join(args.data_root_path, args.dir_3d_output, args.video_name, args.stl_name + '.stl')

    logging.info(f"[main] Loading mesh from {stl_path} ...")
    faces = Mesh.from_file(stl_path).vectors
    faces = torch.from_numpy(faces.copy())

    #------------- log configurations -------------#
    logging.info('\n  - '.join([
        '[main] configuration:',
        f'video_name:                       {args.video_name}',
        f'stl_name:                         {args.stl_name}',
        f'face_n:                           {faces.shape[0]}',
        f'texture_map_max_sizeK:            {texture_map_max_sizeK}',
        f'texture_map_min_triangle_side:    {args.texture_map_min_triangle_side}',
        f'total_video_frames_count:         {total_video_frames_count}',
        f'frame_idx_list:                   {frame_idx_list_final}',
    ]) + '\n')

    timestamps.append(toc('0. Initialize'))

    tic()

    #------------- build an EqtriangleTiler instance -------------#
    tiler = EqtriangleTiler(texture_map_max_sizeK=texture_map_max_sizeK,
                            texture_map_min_triangle_side=args.texture_map_min_triangle_side,
                            faces=faces,
                            K=args.texture_merge_topK,
                            device=device)

    #------------- generate single mesh obj file -------------#
    merged_obj_path = f'{output_path_prefix}_merged.obj'
    merged_png_path = merged_obj_path[:-4] + '.png'
    if not os.path.exists(merged_obj_path):
        tiler.save_obj_with_texture(merged_obj_path)

    timestamps.append(toc('1. Write an obj file'))

    merged_png_path_thisstep = '_'.join([
        output_path_prefix,
        f'merged{args.texture_frame_idx_list_step}',
        f'top{args.texture_merge_topK}outlier',
        f'colorthreshold{args.texture_merge_topKoutlier_colorthreshold}'
    ]) + '.png'

    #------------- render perspective of textured mesh -------------#
    texture_map_list = [f'{output_path_prefix}_cam{frame_i:04d}.png' for frame_i in frame_idx_list_final]
    texture_map_list = [merged_png_path_thisstep] + texture_map_list

    #------------- depth map (equirectangular) generation (both raft (resized) and rendered (from mesh) -------------#
    tic()

    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, first_video_frame = video.read()
    assert (success)

    mesh_near = 0.0001
    mesh_far = args.result_rendering_farplane

    depth_renderer = EquirectDepthRenderer(faces, mesh_near, mesh_far, device, cubeside=1024)
    raft_depth_dir = os.path.join(args.data_root_path, args.dir_depth_output, args.video_name + args.input_depth_dir_postfix)
    raft_depth_paths = [os.path.join(raft_depth_dir, f'{frame_id}.exr') for frame_id in frame_idx_list_final]
    perturbation_kernel_size_half = depth_renderer.cubeside // 50
    visibility_computer = functools.partial(
        render_depth_and_compute_visibility,
        depth_renderer,
        perturbation_kernel_1d=get_gaussian_kernel_separated(perturbation_kernel_size_half)
    )

    gc.collect()
    torch.cuda.empty_cache()
    timestamps.append(toc('2. Render mesh depth map'))

    #------------- render perspective of textured mesh -------------#
    human_mask = torch.from_numpy(cv2.imread(os.path.join(input_video_dir, 'mask_img.png'), -1) > 0.5).to(device)

    background_cubemap_dump_path = os.path.join(
        f'{output_path_prefix}_merged{args.texture_frame_idx_list_step}_background',
        'cubeface_%d.png'
    )
    background_cubemapper = BackgroundCubemapper(
        W=500,
        device=device
    )
    background_loaded = background_cubemapper.load(background_cubemap_dump_path)

    texture_loaded = os.path.exists(merged_png_path_thisstep)

    #------------- texture merge  -------------#
    if texture_loaded and background_loaded:
        logging.info(f'The merged texture map dump is found: {os.path.basename(merged_png_path_thisstep)}\n'
                     f'Remove this file for re-calculation ...')
    else:
        if not texture_loaded:
            #------------- put topK textures -------------#
            tic()

            tiler.texture_merge_topKoutlier_colorthreshold = args.texture_merge_topKoutlier_colorthreshold
            tiler.texture_merge_topKoutlier_prioritypenalty = args.texture_merge_topKoutlier_prioritypenalty
            tiler.texture_merge_topKoutlier_batchsizeM = int(args.texture_merge_topKoutlier_batchsizeM * (10**6))

            topKoutlier_dump_dir = f'{output_path_prefix}_merged{args.texture_frame_idx_list_step}_topK_dump'

            texture_topKdump_loaded = tiler.load(topKoutlier_dump_dir)
            gc.collect()
            torch.cuda.empty_cache()
        else:
            texture_topKdump_loaded = True

        logging.info(f"[main] texture_loaded: {texture_loaded}, texture_topKdump_loaded: {texture_topKdump_loaded}, background_loaded: {background_loaded}")

        if not (texture_topKdump_loaded and background_loaded):

            if not texture_topKdump_loaded:
                #------------- calculate frames priority for each face -------------#
                logging.info("[main] Calculating frame scores  for each face ...")

                score_function = functools.partial(score_function_linear_sum, weight_disp=1, weight_angle=0)
                frame_scores = calculate_frame_score_per_face(faces.to(device),
                                                              extrinsics[frame_idx_list_final],
                                                              visibility_computer,
                                                              raft_depth_paths,
                                                              score_function,
                                                              human_mask)
                gc.collect()
                torch.cuda.empty_cache()
                logging.info("[main] Calculating topKoutlier frames for each face ...")
                frame_scores = calculate_topKoutlier_frames_per_face(frame_scores, args.texture_merge_topK)

            gc.collect()
            torch.cuda.empty_cache()

            #------------- generate multiple texture png file (for each frame) -------------#
            for frame_th, frame_i in enumerate(frame_idx_list_final):
                logging.info(f'[main] Warp texture from frame {frame_i}')

                texture_sampler = EquirectTextureSampler.from_file(
                    video,
                    frame_i,
                    human_mask,
                    device,
                    visibility_map=-(frame_scores[frame_th, tiler.facei_map.clamp(min=0)] < 0).float()
                )

                if not texture_topKdump_loaded:
                    tiler.put_texture_map(
                        texture_sampler.sample_from_xyz_map(
                            xyz_map=tiler.xyz_map,
                            RT=extrinsics[frame_i],
                            mask=tiler.facei_map >= 0,
                        ),
                        my_score=frame_scores[frame_th, :]
                    )

                if not background_loaded:
                    pass

                gc.collect()
                torch.cuda.empty_cache()

            #------------- save topK textures before merge -------------#
            if not texture_topKdump_loaded:
                logging.info(f'[main] Saving topK dumpfile: {topKoutlier_dump_dir}')
                tiler.dump(topKoutlier_dump_dir)

            if not background_loaded:
                logging.info(f'[main] Saving background_cubemap dumpfile: {background_cubemap_dump_path}')
                background_cubemapper.dump(background_cubemap_dump_path)

        timestamps.append(toc('3. Gather topK textures'))

        if not texture_loaded:
            #------------- save merged texture after inpainting (reduce effect of aliasing) -------------#
            tic()

            inpainted_texture_map = tiler.inpaint_texture_map()
            timestamps.append(toc('4. Outlier removal among topK'))

            logging.info(f'[main] Saving merged texture ...')
            cv2.imwrite(merged_png_path, inpainted_texture_map)
            cv2.imwrite(merged_png_path_thisstep, inpainted_texture_map)
            cv2.imwrite(merged_png_path_thisstep[:-4] + '_mask.png', torch2np(tiler.texture_mask.clamp(max=1)*255))

    total_elapsed_time = sum([stamp[1] for stamp in timestamps])
    logging.info(
        '\n\t\t'.join(['[main] Finished...'] + [f'%-40s%10.2f sec' % stamp for stamp in timestamps] + ['  >> Total: %.2f sec' % total_elapsed_time])
    )

    if not args.result_rendering_skip:
        record_path = None
        if args.result_rendering_record:
            record_path = f'{os.path.dirname(output_path_prefix)}_perspective_{args.result_rendering_trajectory_name}.avi'

        texture_map_provider = TextureMapProviderEquitriangleTiler(
            video,
            [-1] + list(frame_idx_list_final),
            texture_map_list,
            torch.cat([torch.eye(4).float().to(extrinsics.device)[None, ...], extrinsics[frame_idx_list_final]], dim=0),
            tiler.xyz_map,
            tiler.facei_map,
            human_mask,
            visibility_computer=functools.partial(visibility_computer, visualize=False),
            raft_depth_paths=[''] + raft_depth_paths
        )

        use_traj_render = True
        if use_traj_render:
            _, rendering_extrinsics = load_extrinsics_from_csv(
                os.path.join(input_video_dir, args.result_rendering_trajectory_name + ".csv"),
                args.max_video_frames_count
            )
            rendering_translations = extrinsic2translation(rendering_extrinsics)  # CAM * XYZ

            visualize_path=False
            if visualize_path:
                import matplotlib.pyplot as plt
                plt.figure()
                ax = plt.axes(projection='3d')
                plt.xlabel('x')
                plt.ylabel('y')
                ax.scatter3D(*torch2np(rendering_translations.T), c='red')
                ax.scatter3D(*torch2np(extrinsic2translation(extrinsics)[frame_idx_list_final].T), c='blue')
                plt.waitforbuttonpress(0.01)

            camera_matrix_looper = CameraMatrixLooper(
                intrinsic=SpecialCameraMatrix.camera2image(
                    args.result_rendering_h, args.result_rendering_w,
                    hfov=args.result_rendering_fov, near=mesh_near, far=mesh_far
                ),
                extrinsic=np.eye(4),
                circular_offset_list=torch2np(rendering_translations),
                enable_autorotate=args.result_rendering_autorotate
            )
        else:
            if args.result_rendering_along_input_traj:
                circular_offset_list = torch2np(extrinsic2translation(extrinsics))
            else:
                circular_offset_list = SpecialCameraMatrix.circular_translation_offset(radius=0.2, n=100)

            camera_matrix_looper = CameraMatrixLooper(
                intrinsic=SpecialCameraMatrix.camera2image(
                    args.result_rendering_h, args.result_rendering_w,
                    hfov=args.result_rendering_fov, near=mesh_near, far=mesh_far
                ),
                extrinsic=texture_map_provider.get_current_extrinsic(),
                circular_offset_list=circular_offset_list,
                enable_autorotate=args.result_rendering_autorotate
            )

        background_cubemap_drawer = framework.OpenglDrawerCubemap(
            texture_map_provider=TextureMapProviderCubeTiler(
                extrinsics=np.eye(4)[None, ...],
                img_list=[background_cubemapper.cubefaces_for_opengl()]
            )
        )

        mesh_drawer = framework.OpenglDrawerTexturedMesh(
            'perspective_rasterize.vert',
            'texture_colors.frag',
            tiler.faces.numel() // 3, 4,
            tiler.faces, tiler.texture_uvcoord,
            texture_map_provider=texture_map_provider
        )

        framework.OpenglRendererArcball('MeshRendering').render(
            args.result_rendering_h, args.result_rendering_w,
            camera_matrix_looper=camera_matrix_looper,
            drawers=[mesh_drawer],
            record_path_format=record_path,
            record_speed=args.result_rendering_record_speed,
            record_one_cycle_only=args.result_rendering_record_one_cycle_only,
            initial_rotation=args.result_rendering_initial_rotation
        )
    return


def score_function_linear_sum(weight_disp, weight_angle, dist=None, angle=None):
    assert (dist.min() > 0.000001)
    score = 0
    denominator = 0

    if dist is not None:
        score += weight_disp * (1 / dist)
        denominator += weight_disp

    if angle is not None:
        score += weight_angle * angle
        denominator += weight_angle

    assert(denominator > 0)
    return score / denominator

def calculate_frame_priority_per_face(scores):
    cam_idx_sorted = torch.argsort(scores, dim=0)

    priority = torch.zeros_like(cam_idx_sorted)
    priority_idx, face_idx = torch.meshgrid(torch._dim_arange(priority, dim=0), torch._dim_arange(priority, dim=1))
    priority[cam_idx_sorted, face_idx] = priority_idx

    return priority

def calculate_topKoutlier_frames_per_face(scores, K):
    _, cam_idx_topk = torch.topk(scores, K, dim=0, largest=True, sorted=True)
    del _
    gc.collect()
    torch.cuda.empty_cache()

    priority_idx, face_idx = torch.meshgrid(
        torch.arange(cam_idx_topk.shape[0], dtype=torch.float32),
        torch.arange(cam_idx_topk.shape[1], dtype=torch.float32)
    )

    face_idx = face_idx.to(torch.long)
    priority_idx = priority_idx.masked_fill(scores[cam_idx_topk, face_idx] < 0, -2)    # close topK but invisible
    never_visible_mask = (scores[cam_idx_topk[0], face_idx[0]] < 0)
    priority_idx[0, face_idx[0, never_visible_mask]] = -1

    scores[:] = -2
    scores[cam_idx_topk, face_idx] = priority_idx
    return scores

def calculate_frame_score_per_face(faces, extrinsics, visibility_computer, raft_depth_paths, score_function, human_mask, keep_invisible=False):
    # faces: FACE * V012 * XYZ
    # extrinsics: CAM * 4 * 4

    cam_batch_size = 50
    cam_n = extrinsics.shape[0]
    scores = []
    for cam_begin in range(0, cam_n, cam_batch_size):
        logging.info(f'[main] calculate_frame_score_per_face ({int(cam_begin / cam_n * 100)}%)')
        cam_end = min(cam_begin + cam_batch_size, cam_n)
        scores_onebatch = calculate_frame_score_per_face_onebatch(
            faces,
            extrinsics[cam_begin:cam_end],
            visibility_computer,
            raft_depth_paths[cam_begin:cam_end],
            score_function,
            human_mask,
            keep_invisible
        )
        scores.append(scores_onebatch)
    scores = torch.cat(scores, dim=0)

    gc.collect()
    torch.cuda.empty_cache()

    return scores


def calculate_frame_score_per_face_onebatch(faces, extrinsics, visibility_computer, raft_depth_paths, score_function, human_mask, keep_invisible):
    # faces: FACE * V012 * XYZ
    # extrinsics: CAM * 4 * 4

    if isinstance(faces, np.ndarray):
        faces = torch.from_numpy(faces.copy()).to(extrinsics.device)

    # ------------- viewing distance score -------------#
    face_centers = faces.mean(axis=1)  # FACE * XYZ
    translations = extrinsic2translation(extrinsics).T.unsqueeze(dim=2)  # XYZ * CAM * 1
    face_to_cam = translations - face_centers.T.unsqueeze(dim=1)  # XYZ * CAM * FACE
    face_to_cam_length = face_to_cam.norm(dim=0)  # CAM * FACE

    viewing_angle_score = True
    if viewing_angle_score:
        # ------------- viewing angle score -------------#
        side01 = faces[:, 0, :] - faces[:, 1, :]  # FACE * XYZ
        side02 = faces[:, 2, :] - faces[:, 1, :]  # FACE * XYZ
        face_normals = torch.cross(side01, side02).T.unsqueeze(dim=1)  # XYZ * 1 * FACE
        face_normals_length = face_normals.norm(dim=0).clamp(min=0.001)  # 1 * FACE
        cosines = torch.abs((face_normals * face_to_cam).sum(dim=0))  # CAM * FACE
        cosines = cosines / (face_to_cam_length * face_normals_length)  # CAM * FACE

        scores = score_function(dist=face_to_cam_length, angle=cosines)
    else:
        scores = score_function(dist=face_to_cam_length)

    # ------------- invisible -------------#
    xys, zs = xyzs2equirectxys(face_centers, human_mask.shape[0], extrinsics)
    xys_long = xys.to(torch.long)
    for i, E in enumerate(extrinsics):
        visibility_weight = visibility_computer(face_centers, E, raft_depth_paths[i])#, visualize=True)
        scores[i] *= visibility_weight
        scores[i, ~human_mask[xys_long[i, ..., 0], xys_long[i, ..., 1]]] = MININF
        gc.collect()
        torch.cuda.empty_cache()

    del xys, zs
    scores_cpu = scores.to('cpu')

    del scores
    torch.cuda.empty_cache()

    return scores_cpu

def generate_visibility_map(depth_renderer, extrinsics, raft_depth_paths, visibility_map_paths):
    os.makedirs(os.path.dirname(visibility_map_paths[0]), exist_ok=True)
    for i, visibility_map_path in enumerate(visibility_map_paths):
        if os.path.exists(visibility_map_path):
            continue

        raft_depth_equirect = torch.from_numpy(cv2.imread(raft_depth_paths[i], -1)).float().to(device)
        raft_depth_equirect[raft_depth_equirect <= 0] = - 1 / MININF
        raft_depth_equirect = 1 / raft_depth_equirect

        depth_renderer.set_extrinsic(extrinsics[i])
        rendered_depth_cubefaces = depth_renderer.depth_maps
        rendered_depth_cubefaces[rendered_depth_cubefaces <= 0] = -MININF
        raft_depth_cubefaces = depth_renderer.resample_equirect_depth_to_cube_depth(raft_depth_equirect)

        cv2.imshow('rendered_depth_cubefaces and raft_depth_cubefaces', 1 / cv2.resize(np.vstack([
            np.hstack(torch2np(rendered_depth_cubefaces)),
            np.hstack(torch2np(raft_depth_cubefaces))
        ]), dsize=None, fx=0.2, fy=0.2))
        cv2.waitKey(10)

def render_depth_and_compute_visibility(depth_renderer, xyzs, extrinsic, raft_depth_path, perturbation_kernel_1d=None, visualize=False):
    # xyzs: N * XYZ
    # perturbation: half viewing angle in radian

    raft_depth_equirect = torch.from_numpy(cv2.imread(raft_depth_path, -1)).float().to(device)
    raft_depth_equirect[raft_depth_equirect <= 0] = 1 / MININF
    raft_depth_equirect = 1 / raft_depth_equirect

    depth_renderer.set_extrinsic(extrinsic)
    rendered_depth_cubefaces = depth_renderer.depth_maps
    rendered_depth_cubefaces[rendered_depth_cubefaces <= 0] = MININF
    raft_depth_cubefaces = depth_renderer.resample_equirect_depth_to_cube_depth(raft_depth_equirect)

    mesh_depth = transform_xyzs(xyzs, extrinsic).norm(dim=-1)
    sampled_rendered_depth = depth_renderer.sample_depth_from_xyzs(xyzs)
    if sampled_rendered_depth == None:
        return torch.zeros_like(mesh_depth)

    sampled_raft_depth = depth_renderer.sample_depth_from_xyzs(xyzs, depth_maps=raft_depth_cubefaces)
    if perturbation_kernel_1d is None:
        return visibility_function(mesh_depth, sampled_rendered_depth, sampled_raft_depth)
    else:
        match_between_raft_and_rendered_cubefaces = ((raft_depth_cubefaces / rendered_depth_cubefaces - 1).abs() <= 0.1).float()
        edge_aware_match_between_raft_and_rendered_cubefaces = torch.min(
            match_between_raft_and_rendered_cubefaces,
            convolve_images(match_between_raft_and_rendered_cubefaces, perturbation_kernel_1d)
        )
        if visualize:

            cv2.imshow('raft / rendered', cv2.resize(np.vstack([
                np.hstack(torch2np(map)) for map in [
                    match_between_raft_and_rendered_cubefaces,
                    convolve_images(match_between_raft_and_rendered_cubefaces, perturbation_kernel_1d),
                    edge_aware_match_between_raft_and_rendered_cubefaces
                ]
            ]).clip(0, 1), dsize=None, fx=0.3, fy=0.3))
            cv2.waitKey(100)

        sampled_mesh_rendered_depth_ratio = mesh_depth / sampled_rendered_depth
        visibility_from_rendered_depth = (sampled_mesh_rendered_depth_ratio <= 1 + visibility_depth_eps)
        sampled_edge_aware_match_between_raft_and_rendered = depth_renderer.sample_depth_from_xyzs(xyzs, depth_maps=edge_aware_match_between_raft_and_rendered_cubefaces)

        sampled_edge_aware_match_between_raft_and_rendered[~visibility_from_rendered_depth] = (sampled_edge_aware_match_between_raft_and_rendered[~visibility_from_rendered_depth] - 2) * sampled_mesh_rendered_depth_ratio[~visibility_from_rendered_depth]

        return sampled_edge_aware_match_between_raft_and_rendered

def convolve_images(imgs, kernel):
    cubeside = imgs.shape[-1]
    convolved = torch.conv1d(
        imgs.reshape(-1, 1, cubeside),
        kernel[None, ...],
        padding=[kernel.shape[-1] // 2]
    ).reshape(6, -1, cubeside)
    convolved = torch.conv1d(
        convolved.transpose(-1, -2).reshape(-1, 1, cubeside),
        kernel[None, ...],
        padding=[kernel.shape[-1] // 2]
    ).reshape(6, -1, cubeside).transpose(-1, -2)
    return convolved