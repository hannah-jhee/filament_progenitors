// g++ -O3 -Wall -shared -std=c++14 -fPIC -shared pick_parts.cpp -o picklib.so -fopenmp
#include <iostream>
#include <cmath>
#include <omp.h>
#include <string>

using namespace std;

extern "C"{
void function(float* particles, float* halos, int prow, int hrow, int snapNum){
    float dist, px, py, pz, hx, hy, hz, hrvir, hmass;
    float dx,dy,dz; // delta x
    float* answer = new float[prow]();
    //bool INSIDE=false;
    //float buffer[3];
    string filename = "./particles_" + to_string(snapNum) + ".bin";
    cout << filename << endl;

#pragma omp parallel for private(px, py, pz, hx, hy, hz, hrvir, dx, dy, dz, dist)
    for (int i=0; i<prow; i++) {
        
        answer[i] = 0.;

        if (i%1000000==0) {
            cout << i << endl;
        }

        px = particles[i*3];
        py = particles[i*3+1];
        pz = particles[i*3+2];

        
        for (int j=0; j<hrow; j++) {
            hx = halos[j*5];
            hy = halos[j*5+1];
            hz = halos[j*5+2];
            hrvir = halos[j*5+3];
            hmass = halos[j*5+4];

            dx = min( abs(px-hx), (float)120.-abs(px-hx) );
            dy = min( abs(py-hy), (float)120.-abs(py-hy) );
            dz = min( abs(pz-hz), (float)120.-abs(pz-hz) );

            dist = dx*dx + dy*dy + dz*dz;

            if (dist < hrvir*hrvir) {
                answer[i] = hmass;
                break;
            }
        }
    }
   
    
    FILE* fp = fopen(filename.c_str(), "wb");
    if (fp) {
        fwrite(answer, sizeof(float), prow, fp);
        fclose(fp);
    } else {
        perror("Error opening file");
    }

    delete[] answer;
    return;
}}