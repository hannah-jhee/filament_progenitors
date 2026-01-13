// gcc -O3 -Wall            -fPIC -shared faster_works.c -o fastlib_mem.so -fopenmp
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int particles_in_cylinder(int* answer, float* particles, int prow, float* center, float* normal, float L, float R) {
    /*
    answer : idx to fill out
    particles : particle coordinates
    prow : # of particles
    center : filament center
    normal : filament parallel vector
    L : filament length to include
    R : radial extent
    */
    float px,py,pz,prj,dist;
    float prj_vec[3];

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

    return 0;
}


int particles_in_flexible_cylinder(int* answer_, float* pdist_, float* particles, int prow, float* filaments, int frow, float R) {
    /*
    not used
    */
    float px,py,pz, fx,fy,fz;
    float dist, min_dist;
    int min_idx;

    #pragma omp parallel for private(px,py,pz,fx,fy,fz,dist,min_dist,min_idx)
    for (int i=0; i<prow; i++){
        answer_[i] = 0;
        pdist_[i] = -999.;

        px = particles[i*3];
        py = particles[i*3+1];
        pz = particles[i*3+2];
        
        min_dist = 99.;
        min_idx = -99;
        for (int j=0; j<frow; j++) {
            fx = filaments[j*3];
            fy = filaments[j*3+1];
            fz = filaments[j*3+2];

            dist = (px-fx)*(px-fx) + (py-fy)*(py-fy) + (pz-fz)*(pz-fz);
            if (dist<min_dist) {
                min_dist=dist;
                min_idx = j;
            }
        }
        if ((min_idx!=0) && (min_idx!=frow-1) && (sqrt(min_dist)<=R)) {
            answer_[i] = 1;
            pdist_[i] = sqrt(min_dist);
        }
    }

    return 0;
}

int particle_dist_in_flexible_cylinder(float* pdist_, float* particles, int prow, float* filaments, int frow) {
    /*
    pdist_ : distance from a particle to a filament
    particles : particle coordinates
    prow : # of particles
    filaments : filament segments coordinates
    frow : # of filament segments
    */
    float px,py,pz, fx,fy,fz;
    float dist, min_dist;
    int min_idx;

    #pragma omp parallel for private(px,py,pz,fx,fy,fz,dist,min_dist)
    for (int i=0; i<prow; i++){
        //pdist_[i] = -999.;

        px = particles[i*3];
        py = particles[i*3+1];
        pz = particles[i*3+2];
        
        min_dist = 999.;
        min_idx = 0;
        for (int j=0; j<frow; j++) {
            fx = filaments[j*3];
            fy = filaments[j*3+1];
            fz = filaments[j*3+2];
            
            dist = (px-fx)*(px-fx) + (py-fy)*(py-fy) + (pz-fz)*(pz-fz);
            if (dist<min_dist) {
                min_dist=dist;
                min_idx = j;
            }

        }
        if ((min_idx!=0) && (min_idx!=frow-1))
            pdist_[i] = sqrt(min_dist);
        //if (i==0 || i==1) 
        //    printf("%d, %d: (%g, %g, %g), (%g, %g, %g) -> %g\n", i, min_idx, px, py, pz, filaments[min_idx*3], filaments[min_idx*3+1], filaments[min_idx*3+2], sqrt(min_dist));
        
    }

    return 0;
}
/*
int calculate_radial_profile(int snapNum, float* Rbins, float* fcoords, float* hcoords, float* _density) {
    char filename[256];
    snprintf(filename, sizeof(filename), "/data2/Ncluster/particle/particles_pos_%03d.bin", snapNum);

    // read particle file
    FILE* fp = fopen(filename, "rb");
    if (fp == NULL) {
        perror("Failed to load particle file\n");
        return EXIT_FAILURE;
    }

    // get the number of particles and save the particle coordinates
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    rewind(fp);
    size_t n_particles = file_size / sizeof(float);
    float* pcoords = (float *)malloc(file_size);
    
    if (fread(pcoords, sizeof(float), n_particles, fp) != n_particles) {
        perror("Failed reading the data...");
        free(pcoords);
        fclose(fp);
        return EXIT_FAILURE;
    }
    
    // loop over particles    
    float xmin,xmax, ymin,ymax, zmin,zmax;
    float Rmin,Rmax;
    float px,py,pz;

    for (int i=0; i<n_particles; i++) {
        px = pcoords[i*3];
        py = pcoords[i*3+1];
        pz = pcoords[i*3+2];

        if ((px>=xmin-10) && (py<=xmax+10) && (py>=ymin-10) && (py<=ymax+10) && (pz>=zmin-10) && (pz<=zmax+10)) { // this is rough cut
            
        }
    }



    free(pcoords);

    return 0;
}

int main() {
    calculate_radial_profile(168);
    return 0;
}*/