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
    images.push_back(imread("image/kirby.jpg"));     // �J�[�r�B
    images.push_back(imread("image/uvChecker.png")); // UV
    images.push_back(imread("image/map.png"));       // �}�b�v�`�b�v

    // ���ׂẲ摜���������ǂݍ��܂�Ă��邱�Ƃ��m�F
    for (size_t i = 0; i < images.size(); ++i) {
        if (images[i].empty()) {
            cout << "�摜 " << i << " ���J�����Ƃ��ł��܂���ł����B" << endl;
            return -1;
        }
    }

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

    // �e�摜�ɂ��ē����𒊏o
    for (const Mat& image : images) {
        Mat grayImage;
        cvtColor(image, grayImage, COLOR_BGR2GRAY);

        vector<KeyPoint> keypoints;
        Mat imageDescriptors;
        sift->detectAndCompute(grayImage, noArray(), keypoints, imageDescriptors);

        // ���̉摜����̃L�[�|�C���g�ƋL�q�q��ǉ�
        allKeypoints.insert(allKeypoints.end(), keypoints.begin(), keypoints.end());

        if (descriptors.empty()) {
            descriptors = imageDescriptors.clone();
        }
        else {
            vconcat(descriptors, imageDescriptors, descriptors);
        }
    }

    // �V�����摜�p�̋�̃L�����o�X���쐬
    Mat newImage = Mat::zeros(images[0].size(), images[0].type());

    // �����_���ɐV�����摜�ɃL�[�|�C���g��`��
    for (const KeyPoint& kp : allKeypoints) {
        Point center(cvRound(kp.pt.x), cvRound(kp.pt.y));
        int radius = cvRound(kp.size / 2);
        circle(newImage, center, radius, Scalar::all(255), 1, LINE_AA);
    }

    return newImage;
}