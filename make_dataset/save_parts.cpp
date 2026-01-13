// g++ -O3 -Wall -shared -std=c++14 -fPIC -shared save_parts.cpp -o savelib.so -fopenmp
#include <iostream>
#include <cmath>
#include <omp.h>
#include <string>

using namespace std;

extern "C"{
void function(float* particles, int prow, int snapNum){
    string filename = "./particles_pos_" + to_string(snapNum) + ".bin";
    cout << filename << endl;

    FILE* fp = fopen(filename.c_str(), "wb");
    if (fp) {
        fwrite(particles, sizeof(float), prow, fp);
        fclose(fp);
    } else {
        perror("Error opening file");
    }

    return;
}}
