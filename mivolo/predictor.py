# mivolo/predictor.py
from collections import defaultdict
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
import tqdm

# יבוא של המודל וה-detector החדש, ומבנה הנתונים המעודכן
from mivolo.model.mi_volo import MiVOLO
# from mivolo.model.yolo_detector import Detector # הסרנו את היבוא הישן
from mivolo.model.florence2_detector import Florence2Detector # יבוא חדש
from mivolo.structures import AGE_GENDER_INFO, PersonAndFaceResult # שינינו טיפוס


class Predictor:
    def __init__(self, config, verbose: bool = False):
        # אתחול ה-Detector החדש
        # נצטרך להוסיף פרמטרים רלוונטיים ל-config או להגדיר כאן
        # למשל, model_id של Florence-2 במקום detector_weights
        florence_model_id = getattr(config, 'florence_model_id', 'microsoft/Florence-2-base')
        # self.detector = Detector(config.detector_weights, config.device, verbose=verbose) # קוד ישן
        self.detector = Florence2Detector(
             model_id=florence_model_id,
             device=config.device,
             verbose=verbose
             # ניתן להוסיף פרמטרים נוספים אם צריך
        )

        # אתחול מודל MiVOLO נשאר כמעט זהה
        # הערה: חצי דיוק (half) עשוי לדרוש בדיקה עם Florence-2
        use_half = getattr(config, 'half', True)
        self.age_gender_model = MiVOLO(
            config.checkpoint,
            config.device,
            half=use_half, # בדוק אם זה עובד טוב עם Florence
            use_persons=config.with_persons,
            disable_faces=config.disable_faces,
            verbose=verbose,
        )
        self.draw = config.draw

    def recognize(self, image: np.ndarray) -> Tuple[PersonAndFaceResult, Optional[np.ndarray]]:
        # 1. הפעלת ה-Detector החדש
        # הפונקציה predict של Florence2Detector תחזיר PersonAndFaceResult מעודכן
        detected_objects: PersonAndFaceResult = self.detector.predict(image)

        # 2. הפעלת מודל הגיל/מגדר על התוצאות
        # הקריאה הזו אמורה לעבוד אם PersonAndFaceResult הותאם נכון
        # ו-MiVOLO.predict מסתמך על הממשק של PersonAndFaceResult (כמו collect_crops, set_age וכו')
        self.age_gender_model.predict(image, detected_objects)

        # 3. ציור התוצאות (אם נדרש)
        out_im = None
        if self.draw:
            # הפונקציה plot של PersonAndFaceResult צריכה להיות מותאמת
            out_im = detected_objects.plot(
                 ages=True, genders=True # ודא שהפרמטרים נכונים
            )

        return detected_objects, out_im

    def recognize_video(self, source: str) -> Generator:
        """
        זיהוי אובייקטים, גיל ומגדר בקלט וידאו.
        שים לב: פונקציה זו תצטרך התאמה משמעותית אם רוצים מעקב,
        כיוון ש-Florence-2 הבסיסי אינו תומך במעקב.
        הגרסה כאן תבצע זיהוי על כל פריים בנפרד.
        """
        print("Warning: Running video recognition without tracking. Each frame processed independently.")
        video_capture = cv2.VideoCapture(source)
        if not video_capture.isOpened():
            raise ValueError(f"Failed to open video source {source}")

        # לא נשתמש בהיסטוריית מעקב כרגע
        # detected_objects_history: Dict[int, List[AGE_GENDER_INFO]] = defaultdict(list)

        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
             print("Warning: Cannot get total frame count from video source.")
             pbar = tqdm.tqdm() # אינסופי
        else:
             pbar = tqdm.tqdm(total=total_frames)


        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # הפעלת זיהוי וניתוח גיל/מגדר על הפריים הנוכחי
            # אין שימוש ב-track, רק predict
            try:
                detected_objects, out_frame = self.recognize(frame)

                 # אם הציור הופעל ב-recognize, out_frame יכיל את התמונה עם הציורים
                 # אם לא, frame המקורי ישלח
                yield_frame = out_frame if self.draw and out_frame is not None else frame

                 # החזרת התוצאות הנוכחיות (ללא היסטוריה מצטברת) והפריים המצויר/מקורי
                 # נחזיר את אובייקט התוצאות עצמו במקום היסטוריה
                yield detected_objects, yield_frame

            except Exception as e:
                 print(f"\nError processing frame: {e}")
                 # אפשר להחזיר פריים מקורי או לדלג
                 yield None, frame # החזרת None לתוצאות, ופריים מקורי

            if total_frames > 0:
                 pbar.update(1)

        video_capture.release()
        pbar.close()
        print("Video processing finished.")