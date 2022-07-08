#version 330

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 texcoord;

out vec2 theTexCoord;
out float theDepth;

uniform mat4 perspectiveMatrix;

void main()
{
	gl_Position = perspectiveMatrix * position;
    theDepth = gl_Position.w;
    theTexCoord = texcoord.xy;
}
