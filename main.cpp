#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;

// �����̉摜������������ɐV�����摜�𐶐�����֐�
Mat CreateImages(const vector<Mat>& images);

int main() {
    // �����̉摜���x�N�^�[�ɓǂݍ���
    vector<Mat> images;
    images.push_back(imread("image/map1.png"));//��      
    images.push_back(imread("image/map2.png"));//��      
    images.push_back(imread("image/map3.png"));//��     
    images.push_back(imread("image/map4.png"));//�V�A��     

    // �����̉摜������������ɐV�����摜�𐶐�
    Mat newImage = CreateImages(images);

    // ���̉摜�Ɛ��������摜��\��
    for (size_t i = 0; i < images.size(); ++i) {
        string imageName = "Image " + to_string(i + 1);
        imshow(imageName, images[i]);
    }
    imshow("New Image", newImage);

    waitKey(0); // �E�B���h�E���ŃL�[���͂�ҋ@
    return 0;
}

Mat CreateImages(const vector<Mat>& images) {
    Ptr<SIFT> sift = SIFT::create();
    vector<KeyPoint> allKeypoints;
    Mat descriptors;

    // �e�摜�̐F���`
    vector<Scalar> colors = { Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255), Scalar(255, 255, 0) };

    // �V�����摜�p�̋�̃L�����o�X���쐬�@
    Mat newImage = Mat::zeros(images[0].size(), images[0].type());

    // �e�摜�ɂ��ē����𒊏o
    for (size_t i = 0; i < images.size(); ++i) {
        Mat grayImage;
        cvtColor(images[i], grayImage, COLOR_BGR2GRAY);

        vector<KeyPoint> keypoints;
        Mat imageDescriptors;
        sift->detectAndCompute(grayImage, noArray(), keypoints, imageDescriptors);

        // ���̉摜����̃L�[�|�C���g��`��
        for (const KeyPoint& kp : keypoints) {
            Point center(cvRound(kp.pt.x), cvRound(kp.pt.y));
            int radius = cvRound(kp.size / 2);
            // �摜�ɑΉ�����F���g�p���ē����_��`��
            circle(newImage, center, radius, colors[i % colors.size()], 1, LINE_AA);
        }

        // ���̉摜����̃L�[�|�C���g�ƋL�q�q��ǉ�
        allKeypoints.insert(allKeypoints.end(), keypoints.begin(), keypoints.end());

        if (descriptors.empty()) {
            descriptors = imageDescriptors.clone();
        }
        else {
            vconcat(descriptors, imageDescriptors, descriptors);
        }
    }

    return newImage;
}