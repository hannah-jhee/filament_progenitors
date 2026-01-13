// g++ -O3 -Wall -shared -std=c++14 -fPIC -shared faster_works.cpp -o fastlib2.so -fopenmp
#include <iostream>
#include <cmath>
#include <string>

using namespace std;

extern "C"{
int* particles_in_cylinder(float* particles, int prow, float* center, float* normal, float L, float R){
    float px,py,pz,prj,dist;
    float prj_vec[3];
    int* answer = new int[prow]();


    for (int i=0; i<prow; i++){
        answer[i] = 0;
        // Get particle coordinates that are shifted by the center of the cylinder
        px = particles[i*3] - center[0];
        py = particles[i*3+1] - center[1];
        pz = particles[i*3+2] - center[2];

        // Project the particle onto the normal vector
        prj = px*normal[0] + py*normal[1] + pz*normal[2];

        for (int j=0;j<3;j++){
            prj_vec[j] = prj*normal[j];
        }


        dist = sqrt((px-prj_vec[0])*(px-prj_vec[0]) + (py-prj_vec[1])*(py-prj_vec[1]) + (pz-prj_vec[2])*(pz-prj_vec[2]));

        if ((dist<=R) && (prj>=-L/2) && (prj<=L/2)) {
            answer[i] = 1;
        }
    }

    return answer;
}
}