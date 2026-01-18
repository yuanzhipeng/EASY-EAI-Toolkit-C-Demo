//
// Created by 袁志鹏 on 2026/1/17.
//

#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <disp.h>
#include <camera.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include"ocr.h"

using namespace cv;
using namespace std;

#define INDENT "    "
#define THRESHOLD 0.3 // pixel score threshold
#define BOX_THRESHOLD 0.9 // box score threshold
#define USE_DILATION false // whether to do dilation, true or false
#define DB_UNCLIP_RATIO 1.5 // unclip ratio for poly type
#define CAMERA_WIDTH	720
#define CAMERA_HEIGHT	1280
#define	IMGRATIO	3
#define	IMAGE_SIZE		(CAMERA_WIDTH*CAMERA_HEIGHT*IMGRATIO)

pthread_mutex_t img_lock;
int main()
{
    char *pbuf = NULL;
    int ret = 0;
    Mat input_image, rgb_img;
    Mat algorithm_image;


    //camera init
    ret = rgbcamera_init(CAMERA_WIDTH, CAMERA_HEIGHT, 270);
    if (ret) {
        printf("error: %s, %d\n", __func__, __LINE__);
        goto exit3;
    }

	/* 2、初始化显示 */
	ret = disp_init(CAMERA_WIDTH, CAMERA_HEIGHT); //RGB888 default
	if (ret) {
		printf("disp_init() error func:%s, line:%d\n", __func__, __LINE__);
		goto exit1;
	}

    pbuf = (char *)malloc(IMAGE_SIZE);
    if (!pbuf) {
        printf("error: %s, %d\n", __func__, __LINE__);
        ret = -1;
        goto exit2;
    }

    rknn_app_context_t ocr_det_ctx, ocr_rec_ctx;

    /* OCR算法检测模型&识别模型初始化 */
    ocr_det_init("ocr_det.model", &ocr_det_ctx);
    ocr_rec_init("ocr_ret.model", &ocr_rec_ctx);


    while(1) {
        pthread_mutex_lock(&img_lock);
        ret = rgbcamera_getframe(pbuf);
        if (ret) {
            printf("error: %s, %d\n", __func__, __LINE__);
            pthread_mutex_unlock(&img_lock);
            continue;
        }
        algorithm_image = Mat(CAMERA_HEIGHT, CAMERA_WIDTH, CV_8UC3, pbuf);
        input_image = algorithm_image.clone();
        // 将图像从 BGR 转换为 RGB
        cv::cvtColor(input_image, rgb_img, COLOR_BGR2RGB);
        cv::flip(input_image, input_image, 1);  // 1 表示水平翻转
        pthread_mutex_unlock(&img_lock);

        if (input_image.empty()) {
            cout << "Error: Could not capture image from camera" << endl;
            break;
        }

        cv::imwrite("result_1.jpg", input_image);

        /* OCR算法检测模型运行 */
        ocr_det_result results;
        ocr_det_postprocess_params params;
        params.threshold = THRESHOLD;
        params.box_threshold = BOX_THRESHOLD;
        params.use_dilate = USE_DILATION;
        params.db_score_mode = (char*)"slow";
        params.db_box_type = (char*)"poly";
        params.db_unclip_ratio = DB_UNCLIP_RATIO;

        ret = ocr_det_run(&ocr_det_ctx, rgb_img, &params, &results);
        if (ret != 0) {
            printf("inference_ppocr_rec_model fail! ret=%d\n", ret);
        }

        /* 截取文字信息和画框 */
        printf("DRAWING OBJECT\n");
        for (int i = 0; i < results.count; i++)
        {
            printf("[%d]: [(%d, %d), (%d, %d), (%d, %d), (%d, %d)] %f\n", i,
                results.box[i].left_top.x, results.box[i].left_top.y, results.box[i].right_top.x, results.box[i].right_top.y,
                results.box[i].right_bottom.x, results.box[i].right_bottom.y, results.box[i].left_bottom.x, results.box[i].left_bottom.y,
                results.box[i].score);

            line(input_image, Point(results.box[i].left_top.x, results.box[i].left_top.y), Point(results.box[i].right_top.x, results.box[i].right_top.y),
                 Scalar(0, 255, 0), 1, LINE_AA);
            line(input_image, Point(results.box[i].right_top.x, results.box[i].right_top.y), Point(results.box[i].right_bottom.x, results.box[i].right_bottom.y),
                 Scalar(0, 255, 0), 1, LINE_AA);
            line(input_image, Point(results.box[i].right_bottom.x, results.box[i].right_bottom.y), Point(results.box[i].left_bottom.x, results.box[i].left_bottom.y),
                 Scalar(0, 255, 0), 1, LINE_AA);
            line(input_image, Point(results.box[i].left_bottom.x, results.box[i].left_bottom.y), Point(results.box[i].left_top.x, results.box[i].left_top.y),
                 Scalar(0, 255, 0), 1, LINE_AA);

            cv::Mat rgb_crop_image = GetRotateCropImage(rgb_img, results.box[i]);

            /* OCR算法识别模型运行 */
            ocr_rec_result rec_results;
            ocr_rec_run(&ocr_rec_ctx, rgb_crop_image, &rec_results);

            // print text result
            printf("regconize result: %s, score=%f\n", rec_results.str, rec_results.score);
        }




	    /* 3、提交显示 */
	    disp_commit(input_image.data, IMAGE_SIZE);
    }

    exit1:
        free(pbuf);
    pbuf = NULL;
    exit2:
        //shifang hanshu
        rgbcamera_exit();
    exit3:
        return ret;
}
