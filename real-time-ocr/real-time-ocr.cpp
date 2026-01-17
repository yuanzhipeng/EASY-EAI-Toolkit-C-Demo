//
// Created by 袁志鹏 on 2026/1/17.
//

#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <disp.h>
#include <camera.h>
#include <stdlib.h>

#define CAMERA_WIDTH	720
#define CAMERA_HEIGHT	1280
#define	IMGRATIO	3
#define	IMAGE_SIZE		(CAMERA_WIDTH*CAMERA_HEIGHT*IMGRATIO)

int main()
{
    char *pbuf = NULL;
    int ret = 0;

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


    while(1) {
        ret = rgbcamera_getframe(pbuf);
        if (ret) {
            printf("error: %s, %d\n", __func__, __LINE__);
            goto exit1;
        }

	/* 3、提交显示 */
	disp_commit(pbuf, IMAGE_SIZE);
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
