/*
 *  Copyright (C) 2021 Texas Instruments Incorporated - http://www.ti.com/
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *    Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 *    Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the
 *    distribution.
 *
 *    Neither the name of Texas Instruments Incorporated nor the names of
 *    its contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* Add Third-party headers to use open CV */
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

/* Module headers. */
#include <common/include/post_process_image_segmentation.h>

namespace ti::edgeai::common
{
// add namespaces
using namespace std; 
using namespace cv; 

#define CLIP(X) ( (X) > 255 ? 255 : (X) < 0 ? 0 : X)

// RGB -> YUV
#define RGB2Y(R, G, B) CLIP(( (  66 * (R) + 129 * (G) +  25 * (B) + 128) >> 8) +  16)
#define RGB2U(R, G, B) CLIP(( ( -38 * (R) -  74 * (G) + 112 * (B) + 128) >> 8) + 128)
#define RGB2V(R, G, B) CLIP(( ( 112 * (R) -  94 * (G) -  18 * (B) + 128) >> 8) + 128)

// YUV -> RGB
#define C(Y) ( (Y) - 16  )
#define D(U) ( (U) - 128 )
#define E(V) ( (V) - 128 )

#define YUV2R(Y, U, V) CLIP(( 298 * C(Y)              + 409 * E(V) + 128) >> 8)
#define YUV2G(Y, U, V) CLIP(( 298 * C(Y) - 100 * D(U) - 208 * E(V) + 128) >> 8)
#define YUV2B(Y, U, V) CLIP(( 298 * C(Y) + 516 * D(U)              + 128) >> 8)

#if defined(EDGEAI_ENABLE_OUTPUT_FOR_TEST)
#define INVOKE_BLEND_LOGIC(T)                           \
    blendSegMask(reinterpret_cast<uint8_t*>(frameData), \
                 reinterpret_cast<T*>(buff->data),      \
                 getDebugObj(),                         \
                 m_config.inDataWidth,                  \
                 m_config.inDataHeight,                 \
                 m_config.outDataWidth,                 \
                 m_config.outDataHeight,                \
                 m_config.alpha)
#else
#define INVOKE_BLEND_LOGIC(T)                           \
    blendSegMask(reinterpret_cast<uint8_t*>(frameData), \
                 reinterpret_cast<T*>(buff->data),      \
                 m_config.inDataWidth,                  \
                 m_config.inDataHeight,                 \
                 m_config.outDataWidth,                 \
                 m_config.outDataHeight,                \
                 m_config.alpha)
#endif // defined(EDGEAI_ENABLE_OUTPUT_FOR_TEST)

PostprocessImageSemanticSeg::PostprocessImageSemanticSeg(const PostprocessImageConfig   &config,
                                                         const DebugDumpConfig          &debugConfig):
    PostprocessImage(config,debugConfig)
{
}

/**
 * Use OpenCV to do in-place update of a buffer with post processing content like
 * alpha blending a specific color for each classified pixel. Typically used for
 * semantic segmentation models.
 * Although OpenCV expects BGR data, this function adjusts the color values so that
 * the post processing can be done on a RGB buffer without extra performance impact.
 * For every pixel in input frame, this will find the scaled co-ordinates for a
 * downscaled result and use the color associated with detected class ID.
 *
 * @param frame Original RGB data buffer, where the in-place updates will happen
 * @param classes Reference to a vector of vector of floats representing the output
 *          from an inference API. It should contain 1 vector describing the class ID
 *          detected for that pixel.
 * @returns original frame with some in-place post processing done
 */
template <typename T1, typename T2>
static T1 *blendSegMask(T1         *frame,
                        T2         *classes,
#if defined(EDGEAI_ENABLE_OUTPUT_FOR_TEST)
                        DebugDump  &debugObj,
#endif // defined(EDGEAI_ENABLE_OUTPUT_FOR_TEST)
                        int32_t     inDataWidth,
                        int32_t     inDataHeight,
                        int32_t     outDataWidth,
                        int32_t     outDataHeight,
                        float       alpha)
{
    uint8_t    *ptr;
    uint8_t     a;
    uint8_t     sa;
    uint8_t     r;
    uint8_t     g;
    uint8_t     b;
    uint8_t     r_m;
    uint8_t     g_m;
    uint8_t     b_m;
    int32_t     w;
    int32_t     h;
    int32_t     sw;
    int32_t     sh;
    int32_t     class_id;

    a  = alpha * 255;
    sa = (1 - alpha ) * 255;

#if defined(EDGEAI_ENABLE_OUTPUT_FOR_TEST)
    string output;
#endif // defined(EDGEAI_ENABLE_OUTPUT_FOR_TEST)

      // initialising count of pixel for counting pixel of each class
      int defective_pixel=0;
      int pump_pixel=0;
      int background_pixel=0;

    // Here, (w, h) iterate over frame and (sw, sh) iterate over classes
    for (h = 0; h < outDataHeight; h++)
    {
        sh = (int32_t)(h * inDataHeight / outDataHeight);
        ptr = frame + h * (outDataWidth * 3);

        for (w = 0; w < outDataWidth; w++)
        {
            int32_t index;

            sw = (int32_t)(w * inDataWidth / outDataWidth);

            // Get the RGB values from original image
            r = *(ptr + 0);
            g = *(ptr + 1);
            b = *(ptr + 2);

            // sw and sh are scaled co-ordiates over the results[0] vector
            // Get the color corresponding to class detected at this co-ordinate
            index = (int32_t)(sh * inDataWidth + sw);
            class_id =  classes[index];

            // Here we are getting class label of each pixel
            // Count number of pixcel of each label
            if(class_id==1){
            defective_pixel++;
            }
            else if(class_id==0){
            pump_pixel++;
            }
            else {
            background_pixel++;
            }

            // random color assignment based on class-id's

            // Class_id 1 corresponds to defect lets color these pixel with green (10,255,30) color.
            // assign any random color to other pixel 
            
            if(class_id == 1){
              r_m = 10 ;
              g_m = 255; 
              b_m = 30 ;
            }
            else if(class_id==2){ // coloring background with (220,220,220)
                r_m = 220;
                g_m = 220;
                b_m = 220;
            }
            else if(class_id==0){ // coloring pump with (255,128,128)
                r_m = 255;
                g_m = 128;
                b_m = 128;
            }
            else{
                r_m = 10;
                g_m = 30;
                b_m = 50;
            }

            // Blend the original image with mask value
            *(ptr + 0) = ((r * a) + (r_m * sa)) / 255;
            *(ptr + 1) = ((g * a) + (g_m * sa)) / 255;
            *(ptr + 2) = ((b * a) + (b_m * sa)) / 255;

            ptr += 3;
        }
    }

#if defined(EDGEAI_ENABLE_OUTPUT_FOR_TEST)
    output.append("[ ");
    for (h = 0; h < inDataHeight; h++)
    {
        for (w = 0; w < inDataWidth; w++)
        {
            int32_t index;

            index = (int32_t)(h * inDataHeight + w);
            class_id =  classes[index];
            output.append(std::to_string(class_id) + "  ");
        }
    }
    output.append(" ]");

    /* Dump the output object and then increment the frame number. */
    debugObj.logAndAdvanceFrameNum("%s", output.c_str());
#endif // defined(EDGEAI_ENABLE_OUTPUT_FOR_TEST)

    // Adding Text "Defect Percentage" On the result with the use of putText in open CV
    
    float txtSize = static_cast<float>(outDataWidth)/POSTPROC_DEFAULT_WIDTH;
    //float txtSize =0.02*outDataWidth;
    int   rowSize = 40 * outDataWidth/POSTPROC_DEFAULT_WIDTH;
    Scalar text_color(200, 200, 200);
    
    Mat img = Mat(outDataHeight, outDataWidth, CV_8UC3, frame);
    
    std::string title = "Percentage Defect: " + std::to_string( (float)(defective_pixel*100)/(pump_pixel+defective_pixel));
    putText(img, title.c_str(), Point(5, rowSize),
    FONT_HERSHEY_SIMPLEX, txtSize, Scalar(0, 0, 0), 2);
    
    return frame;
}

void *PostprocessImageSemanticSeg::operator()(void             *frameData,
                                              VecDlTensorPtr   &results)
{
    /* Even though a vector of variants is passed only the first
     * entry is valid.
     */
    auto *buff = results[0];
    void *ret  = frameData;

    if (buff->type == DlInferType_Int8)
    {
        ret = INVOKE_BLEND_LOGIC(int8_t);
    }
    else if (buff->type == DlInferType_UInt8)
    {
        ret = INVOKE_BLEND_LOGIC(uint8_t);
    }
    else if (buff->type == DlInferType_Int16)
    {
        ret = INVOKE_BLEND_LOGIC(int16_t);
    }
    else if (buff->type == DlInferType_UInt16)
    {
        ret = INVOKE_BLEND_LOGIC(uint16_t);
    }
    else if (buff->type == DlInferType_Int32)
    {
        ret = INVOKE_BLEND_LOGIC(int32_t);
    }
    else if (buff->type == DlInferType_UInt32)
    {
        ret = INVOKE_BLEND_LOGIC(uint32_t);
    }
    else if (buff->type == DlInferType_Int64)
    {
        ret = INVOKE_BLEND_LOGIC(int64_t);
    }
    else if (buff->type == DlInferType_Float32)
    {
        ret = INVOKE_BLEND_LOGIC(float);
    }

    return ret;
}

PostprocessImageSemanticSeg::~PostprocessImageSemanticSeg()
{
}

} // namespace ti::edgeai::common

