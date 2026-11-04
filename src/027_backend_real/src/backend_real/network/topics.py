TOPIC_KEY_CAMERA = b"cam"  # t,img: tuple[int,cv2.MatLike]
TOPIC_KEY_ACTUATOR = b"actuator"  # action x,y: tuple[int,int]
TOPIC_KEY_VISION_TRACK_CORNERS = (
    b"vision/track_corners"  # imgpts: (8,2) NDArray[np.float64]
)
TOPIC_KEY_VISION_TRACK_BALL = b"vision/track_ball"  # (1,2) NDArray[np.float64]
TOPIC_KEY_VISION_POSE = b"vision/pose"  # rotvec: (3,) NDArray[np.float64]
TOPIC_KEY_MPU = b"mpu"  # (8,) NDArray[np.float64]: (t, ax, ay, az, gx, gy, gz, temp)
