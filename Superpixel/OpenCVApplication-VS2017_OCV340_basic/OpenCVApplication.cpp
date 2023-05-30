// OpenCVApplication.cpp : Defines the entry point for the console application.

#include "stdafx.h"
#include "common.h"

using namespace std;

Mat sobel = (Mat_<float>(3, 3) <<
	-1, 0, 1,
	-2, 0, 2,
	-1, 0, 1);

Mat img_f; // scaled to [0,1]
Mat img_lab; // converted to LAB colorspace

Mat show;
Mat labels;
Mat dists;

float m; // compactness parameter
float S; // superpixel size

float dx, dy;

vector<Point2f> centers; // superpixel centers

int nx, ny;
int N;
int k; //number of superpixels

Mat superpixelImage;

float dist(Point2i p1, Point2i p2) {

	//D = sqrt((dc/m)^2 + (ds/S)^2)  - Formula distantei

	float spatial_weight = 1.0f / (S * S);
	float color_weight = 1.0f / (m * m);

	// Calculate the spatial distance
	float dx2 = p2.x - p1.x;
	float dy2 = p2.y - p1.y;
	float spatial_dist = spatial_weight * (dx2 * dx2 + dy2 * dy2);

	// Calculate the color distance
	Vec3f c1 = img_lab.at<Vec3f>(p1.y, p1.x);
	Vec3f c2 = img_lab.at<Vec3f>(p2.y, p2.x);
	float dl = c2[0] - c1[0];
	float da = c2[1] - c1[1];
	float db = c2[2] - c1[2];
	float color_dist = color_weight * (dl * dl + da * da + db * db);

	return sqrtf(spatial_dist + color_dist);
}

void calculateNewCenters() {
	vector<Point2f> new_centers(centers.size(), Point2f(0, 0));
	vector<int> counts(centers.size(), 0);

	int h = img_lab.rows;
	int w = img_lab.cols;

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {

			//get label pixel(i,j)
			int label = labels.at<int>(i, j);
			Point2f p(j, i);
			if (label < new_centers.size()) {
				new_centers[label] += p;
			}
			if (label < counts.size()) {
				counts[label]++;
			}
		}
	}

	for (int i = 0; i < centers.size(); i++) {
		if (counts[i] > 0) {
			new_centers[i] /= counts[i];
		}
	}

	centers = new_centers;
}


void drawCenters(Mat img) {
	Mat centersImg = Mat::zeros(img.size(), img.type()); // create a blank image

	for (Point center : centers) {
		circle(centersImg, center, 3, Scalar(0, 255, 0), -1); // draw a small circle at each center
	}

	imshow("Centers", centersImg); // show the image
}

void drawCenters2(Mat img) {
	Mat centersImg2 = Mat::zeros(img.size(), img.type()); // create a blank image

	for (Point center : centers) {
		circle(centersImg2, center, 3, Scalar(0, 255, 0), -1); // draw a small circle at each center
	}

	imshow("Centers after update", centersImg2); // show the image
}

void calculateSuperpixels(Mat img) {

	//convert [0,1] interval
	img.convertTo(img_f, CV_32F, 1 / 255.);

	//LAB from RGB
	cvtColor(img_f, img_lab, CV_BGR2Lab);

	int w = img.cols;
	int h = img.rows;

	// Initialize superpixel centers
	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++) {

			// dx x dy - superpixel size
			//dx / 2 - centru superpixel pe axa x  +  j * dx - step
			//dy / 2 - centru superpixel pe axa y   + i * dy - step
			centers.push_back(Point2f(j * dx + dx / 2, i * dy + dy / 2));
		}
	}

	//show centers in Image
	drawCenters(img_lab);

	//initialization
	labels = -1 * Mat::ones(img_lab.size(), CV_32S);
	dists = -1 * Mat::ones(img_lab.size(), CV_32F);

	Mat window; //smaller image cut around center of a superpixel
	Point2f p1, p2;

	// Iterate the algorithm, in tis case iterate 10 times
	for (int ii = 0; ii < 10; ii++) {
		//for each center
		for (int c = 0; c < centers.size(); c++) {
			
			Point2f center = centers[c];
			p1 = center;

			//xmin - pozitia x start - stanga - distanta S fata de centru
			//ymin - pozitia y start - sus - distanta S fata de centru
			//xmax - pozitia x end - dreapta - distanta S fata de centru
			//ymax - pozitia y end - jos - distanta S fata de centru
			int xmin = max<int>(p1.x - S, 0);
			int ymin = max<int>(p1.y - S, 0);
			int xmax = min<int>(p1.x + S, w - 1);
			int ymax = min<int>(p1.y + S, h - 1);

			// Search in a window around the center size 2S X 2S
			window = img_f(Range(ymin, ymax), Range(xmin, xmax));

			// assign pixels to nearest center
			for (int i = 0; i < window.rows; i++) {
				for (int j = 0; j < window.cols; j++) {
					//fiecare pixel din window
					p2 = Point2i(xmin + j, ymin + i);
					//calculam distanta in functie de culoare si spatiu
					float d = dist(p1, p2);
					//stocam distanta anterioara. Din iteratia anterioara
					float last_d = dists.at<float>(p2.y, p2.x);

					//Daca distanta e mai mica -> Update.
					if (d < last_d || last_d == -1) {
						dists.at<float>(p2.y, p2.x) = d;
						labels.at<int>(p2.y, p2.x) = c;
					}
				}
			}
		}
		//Compute new superpixel centers
		calculateNewCenters();
	}
}


void calculateBoundaries(Mat& grad)
{

	Mat labels_float;
	labels.convertTo(labels_float, CV_32F);

	Mat gx, gy;
	filter2D(labels_float, gx, -1, sobel); //gradientul de intensitate in directia x
	filter2D(labels_float, gy, -1, sobel.t()); //gradientul de intensitate in directia y
	magnitude(gx, gy, grad); //magnitudinea gradientului. Locurile cu schimbari mari de intensitate
	grad = (grad > 1e-4);  //binarizare
	grad.convertTo(grad, CV_8U); 
	imwrite("boundaries.png", grad * 255);
	imshow("boundaries.png", grad * 255);
}

Mat colorSuperpixels(Mat img) {
	int n = nx * ny;
	//cout << " N = " << n << " " << centers.size();
	vector<Vec3b> avg_colors(n);
	vector<int> num_pixels(n);

	vector<long> b(n), g(n), r(n);

	for (int y = 0; y < labels.rows; y++) {
		for (int x = 0; x < labels.cols; x++) {
			int lbl = labels.at<int>(y, x);

			// Check if lbl is within the valid range.
			if (lbl < 0 || lbl > n) {
				
				labels.at<int>(y, x) = 0;
				lbl = 0;
				continue;
			}

			Vec3b pix = img.at<Vec3b>(y, x);

			b[lbl] += (int)pix[0];
			g[lbl] += (int)pix[1];
			r[lbl] += (int)pix[2];

			++num_pixels[lbl];
		}
	}

	for (int i = 0; i < n; ++i) {
		int num = num_pixels[i];
		if (num > 0) {
			avg_colors[i] = Vec3b(b[i] / num, g[i] / num, r[i] / num);
		}
	}

	Mat output = img.clone();

	for (int y = 0; y < output.rows; y++) {
		for (int x = 0; x < output.cols; x++) {
			int lbl = labels.at<int>(y, x);
			if (lbl >= 0 && lbl < n && num_pixels[lbl] > 0) {
				output.at<Vec3b>(y, x) = avg_colors[lbl];
			}
		}
	}

	return output;
}


int main()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {

		Mat img = imread(fname);
		imshow("Image", img);

		N = img.cols * img.rows; //nr pixels
		k = 200; // nr of superpixels
		m = 15.0; // factor, pentru distanta
		S = sqrtf(N / float(k)); //superpixel size

		nx = img.cols / S; //superpixel x-dir 
		ny = img.rows / S; //superpixel y-dir
		
						   
		//dimensiunile superpixeli(width, height).		   
		dx = img.cols / float(nx); 
		dy = img.rows / float(ny);

		//aplicam blur pe imaginea initiala
		blur(img, img, Size(3, 3));
		imshow("Blurred", img);


		//calculate superpixels function
		calculateSuperpixels(img);

		//creating boundaries image
		Mat bound;
		calculateBoundaries(bound);

		// visualisation of superpixels by average color
		superpixelImage = colorSuperpixels(img);
		imshow("Super Pixel Color", superpixelImage);
		imwrite("superpixel.jpg", superpixelImage);

		//draw centers after all iterations
		drawCenters2(img);

		waitKey();
	}

	return 0;
}