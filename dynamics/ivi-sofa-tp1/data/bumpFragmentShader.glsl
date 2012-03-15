varying vec3 lightVec,lightVec2;
varying vec3 eyeVec;
varying vec2 texCoord;
uniform sampler2D colorMap;
uniform sampler2D normalMap;

void main (void)
{
    //float distSqr = dot(lightVec, lightVec);
    vec3 lVec = normalize(lightVec); //lightVec * inversesqrt(distSqr);
    vec3 lVec2 = normalize(lightVec2);
    //gl_FragColor = vec4(lVec.z*0.5+0.5,0,0,1);    return;

    vec3 vVec = normalize(eyeVec);
    
    vec4 base = texture2D(colorMap, texCoord);
    //gl_FragColor = base;    return;
    
    vec3 bump = normalize( texture2D(normalMap, texCoord).xyz - 0.5 );
    //gl_FragColor = vec4(bump,1);    return;

    //vec4 vAmbient = vec4(0.2,0.2,0.2,0.0);
    //vec4 vAmbient = gl_FrontLightModelProduct.sceneColor;
    vec4 vAmbient = gl_LightModel.ambient;

    float diffuse1 = max( dot(lVec, bump), 0.0 );
    float diffuse2 = max( dot(lVec2, bump), 0.0 );
    //gl_FragColor = vec4(diffuse1, diffuse2, 1, 1);    return;
    
    float specular1 = (diffuse1 <= 0.0) ? 0.0 : pow(clamp(dot(reflect(-lVec, bump), vVec), 0.0, 1.0), 
                      20.0 );
    float specular2 = (diffuse2 <= 0.0) ? 0.0 : pow(clamp(dot(reflect(-lVec2, bump), vVec), 0.0, 1.0), 
                      100.0 );
    
    vec4 vDiffuse = (gl_LightSource[0].diffuse * diffuse1 + gl_LightSource[1].diffuse * diffuse2);
    //vec4 vDiffuse = (/*vec4(1,1,1,1)*/ gl_FrontLightProduct[0].diffuse * diffuse1 + gl_FrontLightProduct[1].diffuse * diffuse2);

    vec4 vSpecular = vec4((specular1+specular2) * base.a);

    gl_FragColor = ( (vAmbient +vDiffuse )*(base *(1.2+0.3*base.a) ) + 
                     vSpecular
                   );
}
