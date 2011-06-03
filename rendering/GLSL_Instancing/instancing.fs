#version 330
precision highp float;

in Fragment
{
    flat vec4 color;
} fragment;

in Vert
{
	vec2 texcoord;
} vert;

uniform sampler2D Diffuse;
uniform float diffuse_alpha;

varying vec3 lightDir,normal,ambient;

out vec4 color;

void main_textured(void)
{
    color = texture2D(Diffuse,vert.texcoord);//fragment.color;
}

void main(void)
{
    vec4 texel = texture2D(Diffuse,vert.texcoord);//fragment.color;
	vec3 ct,cf;
	float intensity,at,af;
	intensity = max(dot(lightDir,normalize(normal)),0.0);
	cf = intensity*vec3(1.0,1.0,1.0);//intensity * (gl_FrontMaterial.diffuse).rgb+ambient;//gl_FrontMaterial.ambient.rgb;
	af = diffuse_alpha;
		
	ct = texel.rgb;
	at = texel.a;
		
	gl_FragColor = vec4(ct * cf, at * af);	
//	color  = vec4(ct * cf, at * af);	
}
