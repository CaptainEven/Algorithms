@ECHO OFF 

REM Parse parameters
if "%1" == "" goto End  REM root dir to be processed
if "%2" == "" goto End  REM GPU index
if "%3" == "" goto End  REM max image size 

goto MyCommand

:MyCommand
echo Parse parameters done, running...

if not exist %1 (
    md %1\dense
    echo %1 not exists, made %1\dense
) 
else (
    rd /s /q %1\dense
    md %1\dense
)

colmap image_undistorter ^
--image_path %1/images ^
--input_path %1/dslr_calibration_undistorted ^
--output_path %1/dense ^
--output_type COLMAP ^
--max_image_size %3

colmap patch_match_stereo ^
--workspace_path %1/dense ^
--workspace_format COLMAP ^
--PatchMatchStereo.geom_consistency true ^
--PatchMatchStereo.gpu_index %2 ^
--PatchMatchStereo.ncc_sigma 0.6 ^
--PatchMatchStereo.window_radius 9 ^
--PatchMatchStereo.max_image_size %3 ^
--PatchMatchStereo.filter false

colmap stereo_fusion ^
--workspace_path %1/dense ^
--workspace_format COLMAP ^
--input_type geometric ^
--output_path %1/dense/fused.ply

goto End

:End
echo End of processing
exit