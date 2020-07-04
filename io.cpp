#include <fstream>
#include <string>

void printPPM(float *t, int size, std::string filename) {
    std::ofstream ofile(filename);
    ofile << "P3\n" << size << " " << size << "\n255\n";

    int *rgb;
    rgb = (int *)malloc((size*size) * 3 * sizeof(int));

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float val = t[i*size + j];
            int off = 3 * (i * size + j);
            if (val <= -25.0) {
                rgb[2 + off] = 255;
            } else if (val >= -25.0 && val < -5) {
                rgb[2 + off] = 255;
                rgb[1 + off] = (int)(255 * ((val + 25) / 20));
            } else if (val >= -5 && val <= 0.0) {
                rgb[1 + off] = 255;
                rgb[2 + off] = (int)(255 * (1.0 - (val + 5) / 5));
            } else if (val > 0.0 && val <= 5) {
                rgb[1 + off] = 255;
                rgb[0 + off] = (int)(255 * ((val) / 5));
            } else if (val > 5 && val < 25.0) {
                rgb[0 + off] = 255;
                rgb[1 + off] = (int)(255 * ((25 - val) / 20));
            } else {
                rgb[0 + off] = 255;
            }
        }
    }
    for (int i = 0; i < 3*size*size; ++i) {
        ofile << rgb[i] << " ";
    }
    ofile.close();
}