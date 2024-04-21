#include <iostream>
#include <mpi.h>
#include <opencv2/opencv.hpp>

using namespace cv;

class MandelbrotRenderer {
private:
    int imageWidth;
    int imageHeight;
    double minX;
    double maxX;
    double minY;
    double maxY;
    int maxIterations;
    Mat fractalImage;

public:
    MandelbrotRenderer(int width, int height, double min_x, double max_x, double min_y, double max_y, int iterations) :
        imageWidth(width), imageHeight(height), minX(min_x), maxX(max_x), minY(min_y), maxY(max_y), maxIterations(iterations),
        fractalImage(height, width, CV_8UC3) {}

    void render() {
        int processRank, numProcesses;
        MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
        MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

        int rowsPerProcess = imageHeight / numProcesses;
        int startRow = processRank * rowsPerProcess;
        int endRow = (processRank == numProcesses - 1) ? imageHeight : startRow + rowsPerProcess;

        for (int y = startRow; y < endRow; y++) {
            for (int x = 0; x < imageWidth; x++) {
                double cr = minX + (maxX - minX) * x / imageWidth;
                double ci = minY + (maxY - minY) * y / imageHeight;
                int iterations = calculateMandelbrotIterations(cr, ci);

                if (iterations == maxIterations) {
                    fractalImage.at<Vec3b>(y, x) = Vec3b(0, 0, 0);  // Black color for points outside the set
                }
                else {
                    fractalImage.at<Vec3b>(y, x) = generateUniqueColor(iterations);
                }
            }
        }

        if (processRank == 0) {
            Mat finalImage(imageHeight, imageWidth, CV_8UC3);
            MPI_Gather(fractalImage.data, imageWidth * rowsPerProcess * 3, MPI_UNSIGNED_CHAR,
                finalImage.data, imageWidth * rowsPerProcess * 3, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

            namedWindow("Unique Mandelbrot Fractal", WINDOW_NORMAL);
            imshow("Unique Mandelbrot Fractal", finalImage);
            waitKey(0);
        }
    }

private:
    int calculateMandelbrotIterations(double cr, double ci) {
        double zReal = 0.0, zImaginary = 0.0;
        int iterations = 0;

        while (iterations < maxIterations && zReal * zReal + zImaginary * zImaginary < 4.0) {
            double temp = zReal * zReal - zImaginary * zImaginary + cr;
            zImaginary = 2.0 * zReal * zImaginary + ci;
            zReal = temp;
            iterations++;
        }

        return iterations;
    }

    Vec3b generateUniqueColor(int n) {
        int r = (n * 50) % 255;
        int g = (n * 30) % 255;
        int b = (n * 20) % 255;
        return Vec3b(r, g, b);
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int imageWidth = 800;
    int imageHeight = 800;
    double minX = -2.0;
    double maxX = 1.0;
    double minY = -1.5;
    double maxY = 1.5;
    int maxIterations = 1000;

    MandelbrotRenderer renderer(imageWidth, imageHeight, minX, maxX, minY, maxY, maxIterations);
    renderer.render();

    MPI_Finalize();

    return 0;
}