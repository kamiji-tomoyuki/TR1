//openCv���C���N���[�h
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

// �������o & �摜����
Mat CreateImages(const Mat& image);

int main() {
	// �摜��ǂݍ���
	Mat A = imread("image/kirby.jpg");     //�J�[�r�B�@
	Mat B = imread("image/uvChecker.png"); //UV
	Mat C = imread("image/map.png");       //�}�b�v�`�b�v

	// �ǂݎ�点��摜
	Mat image = C;

	// �������w�K & �V�����摜 �𐶐�
	Mat newImage = CreateImages(image);

	// ���̉摜 & ������摜 ��`��I�I
	imshow("Image", image);
	imshow("newImage", newImage);

	waitKey(0);//�摜��`�悳���邽�߂ɕK�v
	return 0;
}

Mat CreateImages(const Mat& image) {
	// �O���[�X�P�[���ɕϊ� -> SIFT�̏������ȒP��
	Mat grayImage;
	cvtColor(image, grayImage, COLOR_BGR2GRAY);

	// �����ʂ����߂�
	Ptr<SIFT> sift = SIFT::create();
	vector<KeyPoint> keyPoint;
	Mat descriptors;//�����_
	sift->detectAndCompute(grayImage, noArray(), keyPoint, descriptors);

	// �V�����摜�𐶐��I�I (TR1_1��ڂ͓����_��`�悷��)
	Mat newImage = image.clone();
	drawKeypoints(image, keyPoint, newImage, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	return newImage;
}