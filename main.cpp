#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace cv::ml;
using namespace std;

// �O���b�h�T�C�Y�i�^�C���T�C�Y�j
const int GRID_SIZE = 32;

Mat extractFeatureDescriptors(const Mat& img, Ptr<SIFT> sift);

Mat processMapChips(const Mat& mapImage);

int main() {
    // �}�b�v�`�b�v�̉摜��ǂݍ���
    Mat mapImage = imread("image/map.png");

    // �}�b�v�`�b�v�̉�� & �s��̏o��
    Mat outputMatrix = processMapChips(mapImage);

    // �s��̕\��
    cout << "Output Matrix:" << endl;
    for (int i = 0; i < outputMatrix.rows; i++) {
        for (int j = 0; j < outputMatrix.cols; j++) {
            cout << outputMatrix.at<int>(i, j) << " ";
        }
        cout << endl;
    }

    imshow("Image", mapImage);


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

Mat processMapChips(const Mat& mapImage) {
    // SIFT�I�u�W�F�N�g�̐���
    Ptr<SIFT> sift = SIFT::create();

    // �摜�̃T�C�Y
    int rows = mapImage.rows / GRID_SIZE;
    int cols = mapImage.cols / GRID_SIZE;

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