# mivolo/model/florence2_detector.py
import os
from typing import Union, Dict, List

import numpy as np
import PIL
import torch
from mivolo.structures import PersonAndFaceResult # נצטרך לשנות את PersonAndFaceResult בהמשך
from transformers import AutoProcessor, AutoModelForCausalLM

# בגלל באג אפשרי ב-ultralytics (למרות שכאן לא משתמשים בו ישירות, נשאיר למקרה הצורך)
# os.unsetenv("CUBLAS_WORKSPACE_CONFIG")

class Florence2Detector:
    def __init__(
        self,
        model_id: str = 'microsoft/Florence-2-base',
        device: str = "cuda", # שינוי כאן ל-cuda כברירת מחדל אם יש לך GPU, או השאר cpu
        half: bool = True,
        verbose: bool = False,
        max_new_tokens=1024,
        num_beams=3,
    ):
        self.device = torch.device(device)
        self.half = half and self.device.type != "cpu"
        self.verbose = verbose
        self.model_id = model_id
        self.task_prompt = '<OD>'

        if self.verbose:
            print(f"Initializing Florence-2 detector with model: {model_id}")
            print(f"Target device: {self.device}, Half precision: {self.half}")

        try:
            # 1. טען את המודל קודם (בדרך כלל נטען ל-CPU כברירת מחדל אם אין device_map)
            print("Loading Florence-2 model...")
            model_loaded = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True
                # הסרנו device_map ו-.to() מכאן
            )
            print("Model loaded. Moving to device...")
            # 2. העבר את המודל ל-device הרצוי
            self.model = model_loaded.eval().to(self.device)
            print(f"Model moved to {self.device}.")

            # 3. טען את ה-processor (בדרך כלל לא דורש העברה מפורשת ל-device)
            print("Loading Florence-2 processor...")
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            print("Processor loaded.")

            # 4. (אופציונלי) המרה ל-half precision אם נדרש
            if self.half:
                print("Attempting to convert model to half precision...")
                try:
                     self.model = self.model.half()
                     print("Model converted to half precision.")
                except Exception as e:
                     print(f"Warning: Could not convert model to half precision: {e}. Using float32.")
                     self.half = False # עדכן את הדגל אם ההמרה נכשלה

        except NameError as ne:
             # תפיסת השגיאה הספציפית שהופיעה
             if 'init_empty_weights' in str(ne):
                 print("\nError: 'init_empty_weights' not defined.")
                 print("This often means the 'accelerate' library is missing or incompatible.")
                 print("Please ensure 'accelerate' is installed in your environment:")
                 print("  pip install accelerate")
                 print("Or try upgrading both libraries:")
                 print("  pip install --upgrade transformers accelerate")
             raise # העלה את השגיאה מחדש לאחר ההודעה
        except Exception as e:
             print(f"\nAn error occurred during Florence-2 initialization: {e}")
             raise


        self.generation_params = {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "do_sample": False
        }
        if self.verbose:
            print(f"Florence-2 detector initialized successfully.")

    # ... (שאר הקוד של הקלאס) ...
    def _generate_florence2_results(self, image: Union[np.ndarray, str, "PIL.Image"]) -> Dict:
        # ... (קוד זה נשאר כמעט זהה) ...
        if isinstance(image, np.ndarray):
            if image.shape[2] == 3:
                image = image[:, :, ::-1] # BGR to RGB
            image = PIL.Image.fromarray(image)
        elif isinstance(image, str):
            image = PIL.Image.open(image)

        inputs = self.processor(text=self.task_prompt, images=image, return_tensors="pt").to(self.device)

        # ודא שהקלט באותו דיוק כמו המודל (חשוב אם משתמשים ב-half)
        if self.half:
            # inputs['pixel_values'] = inputs['pixel_values'].half() # בדרך כלל רק ה-pixel_values
            # או אם המודל לא הצליח לעבור ל-half, ודא שהקלט לא half
            if not next(self.model.parameters()).is_cuda or next(self.model.parameters()).dtype != torch.float16:
                 # אם המודל לא באמת חצי דיוק, ודא שהקלט לא
                 inputs['pixel_values'] = inputs['pixel_values'].float()
            else:
                 # אם המודל חצי דיוק, המר את הקלט
                 inputs['pixel_values'] = inputs['pixel_values'].half()


        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                **self.generation_params
            )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        try:
            results = self.processor.post_process_generation(
                generated_text,
                task=self.task_prompt,
                image_size=(image.width, image.height)
            )
        except Exception as e:
            print(f"Error during Florence-2 post-processing: {e}")
            print(f"Generated text was: {generated_text}")
            return {self.task_prompt: {'bboxes': [], 'labels': []}}

        return results



    def predict(self, image: Union[np.ndarray, str, "PIL.Image"]) -> PersonAndFaceResult:
        """ הפעלת זיהוי אובייקטים והמרת התוצאות לפורמט של MiVOLO """
        if self.verbose:
            print("Running Florence-2 Object Detection...")

        florence_results_dict = self._generate_florence2_results(image)

        # חילוץ הנתונים הרלוונטיים מהמילון
        # ודא שהמפתח הנכון קיים
        if self.task_prompt not in florence_results_dict:
             print(f"Warning: Task prompt '{self.task_prompt}' not found in Florence-2 results.")
             detection_results = {'bboxes': [], 'labels': []}
        else:
             detection_results = florence_results_dict[self.task_prompt]


        bboxes = detection_results.get('bboxes', [])
        labels = detection_results.get('labels', [])

        if self.verbose:
            print(f"Florence-2 detected {len(bboxes)} objects.")
            # אופציונלי: הדפסת התוויות שנמצאו
            # from collections import Counter
            # print(f"Labels found: {Counter(labels)}")


        # סינון ראשוני כדי להשאיר רק 'person' ו-'human face'
        # חשוב: הפורמט של התווית עשוי להשתנות, ודא שהוא 'person' ו-'human face'
        # או התאם את המחרוזות בהתאם.
        filtered_bboxes = []
        filtered_labels = []
        original_indices = [] # לשמור אינדקסים מקוריים אם נצטרך למפות חזרה

        # ודא שהתוויות הן אכן המחרוזות שאתה מצפה להן
        person_label = "person"
        face_label = "human face" # או "face" ? בדוק את הפלט של Florence-2

        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
             # ייתכן ש-Florence מחזיר תווית עם אזור, למשל 'person [<bbox>]'
             # ננקה את התווית לפני ההשוואה
             clean_label = label.split('[')[0].strip().lower()
             if clean_label == person_label or clean_label == face_label:
                 filtered_bboxes.append(bbox)
                 filtered_labels.append(clean_label) # שמירת התווית הנקייה
                 original_indices.append(i)

        if self.verbose:
            print(f"Filtered to {len(filtered_bboxes)} persons/faces.")

        # קבלת התמונה המקורית כ-numpy array אם היא לא כבר
        if isinstance(image, PIL.Image.Image):
            # המרה ל-numpy ול-BGR (אם MiVOLO מצפה ל-BGR)
             orig_img_np = np.array(image)
             # אם התמונה המקורית היתה RGB והמודל מצפה ל-BGR, נמיר כאן
             # orig_img_np = cv2.cvtColor(orig_img_np, cv2.COLOR_RGB2BGR)
        elif isinstance(image, str):
             # טעינה עם PIL והמרה
             img_pil = PIL.Image.open(image)
             orig_img_np = np.array(img_pil)
             # כנ"ל, המרה ל-BGR אם צריך
             # orig_img_np = cv2.cvtColor(orig_img_np, cv2.COLOR_RGB2BGR)
        else: # כבר numpy array (נניח BGR מ-cv2)
             orig_img_np = image.copy() # העתקה למניעת שינוי המקור


        # יצירת אובייקט PersonAndFaceResult (נצטרך להתאים אותו בהמשך)
        # כרגע נעביר את הנתונים המסוננים
        # נצטרך לשנות את __init__ של PersonAndFaceResult
        results_obj = PersonAndFaceResult(
            bboxes=filtered_bboxes,
            labels=filtered_labels,
            orig_img=orig_img_np,
            # אופציונלי: ניתן להוסיף ציוני ביטחון אם Florence-2 מספק אותם
            # confs=[1.0] * len(filtered_bboxes) # ברירת מחדל 1.0 אם אין ציון ביטחון
        )

        return results_obj

    # פונקציית Track לא נתמכת ישירות על ידי Florence-2 הבסיסי
    # def track(self, image: Union[np.ndarray, str, "PIL.Image"]) -> PersonAndFaceResult:
    #     # ידרוש שילוב עם אלגוריתם מעקב נפרד
    #     raise NotImplementedError("Tracking is not directly supported by base Florence-2.")