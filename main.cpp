#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;

// �ǂݎ�点��}�b�v�`�b�v�̃T�C�Y
const int GRID_SIZE = 16;
const int MATRIX_SIZE = 25; // �s��̃T�C�Y

Mat extractFeatureDescriptors(const Mat& img, Ptr<SIFT> sift);
Mat LoadMapChips(const Mat& mapImage);

int main() {
    try {
        // �ǂݎ�点��}�b�v�`�b�v�摜
        Mat mapImage = imread("image/map.png");
        if (mapImage.empty()) {
            cerr << "Error: Could not load image." << endl;
            return -1;
        }

        // �摜�̃v���p�e�B��\��
        cout << "Image properties:" << endl;
        cout << "Width: " << mapImage.cols << endl;
        cout << "Height: " << mapImage.rows << endl;
        cout << "Channels: " << mapImage.channels() << endl;

        if (mapImage.cols < GRID_SIZE * MATRIX_SIZE || mapImage.rows < GRID_SIZE * MATRIX_SIZE) {
            cerr << "Error: Image size is too small for the desired grid size." << endl;
            return -1;
        }

        // �}�b�v�`�b�v����� & �s��𒊏o
        Mat outputMatrix = LoadMapChips(mapImage);

        // �s���\��
        cout << "Output Matrix:" << endl;
        for (int i = 0; i < outputMatrix.rows; i++) {
            for (int j = 0; j < outputMatrix.cols; j++) {
                cout << outputMatrix.at<int>(i, j) << " ";
            }
            cout << endl;
        }

        // ���̉摜��`��
        imshow("Image", mapImage);

        waitKey(0); // �摜��`�悳���邽�߂ɕK�v
    }
    catch (const cv::Exception& e) {
        cerr << "OpenCV Exception: " << e.what() << endl;
    }
    catch (const exception& e) {
        cerr << "Standard Exception: " << e.what() << endl;
    }
    catch (...) {
        cerr << "Unknown Exception occurred." << endl;
    }

    return 0;
}

Mat extractFeatureDescriptors(const Mat& img, Ptr<SIFT> sift) {
    Mat grayImage;
    cvtColor(img, grayImage, COLOR_BGR2GRAY);
    vector<KeyPoint> keypoints;
    Mat descriptors;
    sift->detectAndCompute(grayImage, noArray(), keypoints, descriptors);
    return descriptors;
}

Mat LoadMapChips(const Mat& mapImage) {
    // SIFT�I�u�W�F�N�g�̐���
    Ptr<SIFT> sift = SIFT::create();

    // �摜�̃T�C�Y
    int rows = MATRIX_SIZE; // �s��̍s��
    int cols = MATRIX_SIZE; // �s��̗�

    // �����x�N�g���̊i�[�ꏊ
    Mat allDescriptors;
    vector<int> indices;

    // �e�^�C���̓����x�N�g���𒊏o
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Rect roi(j * GRID_SIZE, i * GRID_SIZE, GRID_SIZE, GRID_SIZE);
            Mat tile = mapImage(roi);
            Mat descriptors = extractFeatureDescriptors(tile, sift);
            if (!descriptors.empty()) {
                allDescriptors.push_back(descriptors);
                indices.push_back(i * cols + j);
            }
        }
    }

    // K-means�N���X�^�����O
    int K = 10;  // �N���X�^��
    Mat labels;
    Mat centers;
    kmeans(allDescriptors, K, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.1), 3, KMEANS_PP_CENTERS, centers);

    // �s��Ƃ��ďo��
    Mat outputMatrix = Mat::zeros(rows, cols, CV_32S);
    for (size_t i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        int r = idx / cols;
        int c = idx % cols;
        outputMatrix.at<int>(r, c) = labels.at<int>(i);
    }

    return outputMatrix;
}