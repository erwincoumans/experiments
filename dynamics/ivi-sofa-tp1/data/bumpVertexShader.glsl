varying vec3 lightVec,lightVec2; 
varying vec3 eyeVec;
varying vec2 texCoord;
attribute vec3 tangent;

void main(void)
{
	gl_Position = ftransform();
	texCoord = gl_MultiTexCoord0.xy;
	
	vec3 n = normalize(gl_NormalMatrix * gl_Normal);
	vec3 t = normalize(gl_NormalMatrix * tangent);
	vec3 b = cross(n, t);
	
	vec3 vVertex = vec3(gl_ModelViewMatrix * gl_Vertex);
	vec3 tmpVec = gl_LightSource[0].position.xyz - vVertex;

	lightVec.x = dot(tmpVec, t);
	lightVec.y = dot(tmpVec, b);
	lightVec.z = dot(tmpVec, n);
    //lightVec = normalize(lightVec);
    //lightVec = t;

	tmpVec = gl_LightSource[1].position.xyz - vVertex;

	lightVec2.x = dot(tmpVec, t);
	lightVec2.y = dot(tmpVec, b);
	lightVec2.z = dot(tmpVec, n);
    //lightVec2 = normalize(lightVec2);

	tmpVec = -vVertex;
	eyeVec.x = dot(tmpVec, t);
	eyeVec.y = dot(tmpVec, b);
	eyeVec.z = dot(tmpVec, n);
}
