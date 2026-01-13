#include <stdio.h>
#include <stdlib.h>

int main() {
    char filename[256];
    int snapNum;
    scanf("%d", &snapNum);
    snprintf(filename, sizeof(filename), "/data2/Ncluster/particle/particles_pos_%03d.bin", snapNum);
    printf("%s\n",filename);

    FILE* fp = fopen(filename, "rb");
    if (fp == NULL) {
        perror("Failed to load particle file\n");
        return EXIT_FAILURE;
    }

    fseek(fp, 0, SEEK_END); // 파일 포인터를 파일 끝으로 이동
    long fileSize = ftell(fp);
    size_t numElements = fileSize / sizeof(float);
    printf("%lu\n", fileSize);
    printf("%lu\n", numElements);

    fclose(fp);

    return 0;
}
