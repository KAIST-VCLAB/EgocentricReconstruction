#version 330 core

in vec2 theTexCoord;
in float theDepth;

out vec4 outputColor;

uniform int colorDisparity;
uniform sampler2D ourTexture;

void main()
{
    if (colorDisparity == 0)
    {
        outputColor = texture(ourTexture, theTexCoord);
    }
    else
    {
        float theDisparity = 1.0 / theDepth;
        outputColor = texture(ourTexture, theTexCoord);
        outputColor = vec4(theDisparity, theDisparity, theDisparity, 1);
    }
}