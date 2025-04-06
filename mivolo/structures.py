# mivolo/structures.py
import math
import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
# הסרנו יבוא של ultralytics Results
# from ultralytics.engine.results import Results
from ultralytics.utils.plotting import Annotator, colors # עדיין שימושי לציור

from mivolo.data.misc import aggregate_votes_winsorized, assign_faces, box_iou # פונקציות עזר נשארות

# קודם: AGE_GENDER_TYPE = Tuple[float, str]
# נשנה כדי שנוכל לאחסן גם ציון מגדר
AGE_GENDER_INFO = Tuple[Optional[float], Optional[str], Optional[float]] # (age, gender, gender_score)





def calculate_distance_penalty(face_bbox, person_bbox, person_height):
    """ מחשב קנס על בסיס מרחק אנכי בין מרכז הפנים לחלק העליון של הגוף """
    fx, fy, _, _ = (face_bbox[0] + face_bbox[2]) / 2, (face_bbox[1] + face_bbox[3]) / 2
    px, py1, _, py2 = person_bbox[0], person_bbox[1], person_bbox[2], person_bbox[3]
    person_top_center_y = py1 # או py1 + person_height * 0.1 (קצת מתחת לקצה העליון)

    # מרחק אנכי יחסי לגובה האדם
    vertical_dist = abs(fy - person_top_center_y)
    relative_vertical_dist = vertical_dist / person_height if person_height > 0 else float('inf')

    # קנס גדל ככל שהמרחק היחסי גדל
    penalty = min(relative_vertical_dist * 2.0, 1.0) # קנס בין 0 ל-1, מוכפל פי 2 כדי להדגיש
    return penalty # קנס גבוה יותר = התאמה פחות טובה

def is_contained(inner_box, outer_box, tolerance=0.85):
    """ בודק אם inner_box מוכל ברובו ב-outer_box """
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box

    inter_x1 = max(ix1, ox1)
    inter_y1 = max(iy1, oy1)
    inter_x2 = min(ix2, ox2)
    inter_y2 = min(iy2, oy2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    inner_area = (ix2 - ix1) * (iy2 - iy1)

    if inner_area == 0: return False
    containment_ratio = inter_area / inner_area
    return containment_ratio >= tolerance








class PersonAndFaceCrops:
     # ... (קוד הקלאס הזה נשאר ללא שינוי) ...
    def __init__(self):
        # int: index of person along results
        self.crops_persons: Dict[int, np.ndarray] = {}
        # int: index of face along results
        self.crops_faces: Dict[int, np.ndarray] = {}
        # int: index of face along results
        self.crops_faces_wo_body: Dict[int, np.ndarray] = {}
        # int: index of person along results
        self.crops_persons_wo_face: Dict[int, np.ndarray] = {}

    def _add_to_output(
        self, crops: Dict[int, np.ndarray], out_crops: List[np.ndarray], out_crop_inds: List[Optional[int]]
    ):
        inds_to_add = list(crops.keys())
        crops_to_add = list(crops.values())
        out_crops.extend(crops_to_add)
        out_crop_inds.extend(inds_to_add)

    def _get_all_faces(
        self, use_persons: bool, use_faces: bool
    ) -> Tuple[List[Optional[int]], List[Optional[np.ndarray]]]:
        """
        Returns
            if use_persons and use_faces
                faces: faces_with_bodies + faces_without_bodies + [None] * len(crops_persons_wo_face)
            if use_persons and not use_faces
                faces: [None] * n_persons
            if not use_persons and use_faces:
                faces: faces_with_bodies + faces_without_bodies
        """

        def add_none_to_output(faces_inds, faces_crops, num):
            faces_inds.extend([None for _ in range(num)])
            faces_crops.extend([None for _ in range(num)])

        faces_inds: List[Optional[int]] = []
        faces_crops: List[Optional[np.ndarray]] = []

        if not use_faces:
             # צריך לדעת כמה אנשים יש סה"כ
             num_total_persons = len(self.crops_persons) + len(self.crops_persons_wo_face)
             add_none_to_output(faces_inds, faces_crops, num_total_persons)
             return faces_inds, faces_crops

        # סדר הפנים צריך להתאים לסדר הגופות כאשר שניהם קיימים
        # נוסיף תחילה את הפנים שיש להם גוף תואם
        face_inds_with_body = list(self.crops_faces.keys())
        person_inds_with_face = list(self.crops_persons.keys()) # האינדקסים של הגופות כאן

        # מיפוי מאינדקס גוף לאינדקס פנים (צריך להיות קיים מחוץ לקלאס הזה, או שנצטרך לאחסן אותו כאן)
        # נניח שקיבלנו מיפוי כזה אם צריך, או נסדר את הקריאות
        # כרגע נניח שסדר ההוספה של הפנים והגופות ב-collect_crops שומר על ההתאמה

        self._add_to_output(self.crops_faces, faces_crops, faces_inds) # פנים עם גופות
        self._add_to_output(self.crops_faces_wo_body, faces_crops, faces_inds) # פנים בלי גופות

        if use_persons:
            # הוספת None לפנים עבור הגופות שאין להם פנים
            add_none_to_output(faces_inds, faces_crops, len(self.crops_persons_wo_face))

        return faces_inds, faces_crops

    def _get_all_bodies(
        self, use_persons: bool, use_faces: bool
    ) -> Tuple[List[Optional[int]], List[Optional[np.ndarray]]]:
        """
        Returns
            if use_persons and use_faces
                persons: bodies_with_faces + [None] * len(faces_without_bodies) + bodies_without_faces
            if use_persons and not use_faces
                persons: bodies_with_faces + bodies_without_faces
            if not use_persons and use_faces
                persons: [None] * n_faces
        """

        def add_none_to_output(bodies_inds, bodies_crops, num):
            bodies_inds.extend([None for _ in range(num)])
            bodies_crops.extend([None for _ in range(num)])

        bodies_inds: List[Optional[int]] = []
        bodies_crops: List[Optional[np.ndarray]] = []

        if not use_persons:
             num_total_faces = len(self.crops_faces) + len(self.crops_faces_wo_body)
             add_none_to_output(bodies_inds, bodies_crops, num_total_faces)
             return bodies_inds, bodies_crops

        # הוספת הגופות שיש להם פנים
        self._add_to_output(self.crops_persons, bodies_crops, bodies_inds)

        if use_faces:
             # הוספת None לגופות עבור הפנים שאין להם גופות
             add_none_to_output(bodies_inds, bodies_crops, len(self.crops_faces_wo_body))

        # הוספת הגופות שאין להם פנים
        self._add_to_output(self.crops_persons_wo_face, bodies_crops, bodies_inds)

        return bodies_inds, bodies_crops


    def get_faces_with_bodies(self, use_persons: bool, use_faces: bool):
        """
        Return tuples: (indices, crops) for bodies and faces, ensuring alignment.
        """
        bodies_inds, bodies_crops = self._get_all_bodies(use_persons, use_faces)
        faces_inds, faces_crops = self._get_all_faces(use_persons, use_faces)

        # ודא שאורך הרשימות זהה
        assert len(bodies_inds) == len(faces_inds), \
            f"Mismatch in body ({len(bodies_inds)}) and face ({len(faces_inds)}) list lengths"

        # מיון כדי להבטיח שהתאמה בין פנים לגוף נשמרת באותו אינדקס
        # נניח שהמיון צריך להיות לפי האינדקס המקורי של הפנים (או הגוף התואם)
        # זה דורש שהאינדקסים ב-faces_inds ו-bodies_inds יהיו עקביים
        # נדרשת לוגיקה מורכבת יותר כאן אם רוצים מיון מובטח.
        # כרגע נניח שהסדר מהפונקציות הפנימיות מספיק נכון.

        return (bodies_inds, bodies_crops), (faces_inds, faces_crops)


    def save(self, out_dir="output"):
        ind = 0
        os.makedirs(out_dir, exist_ok=True)
        for crop_dict_name, crops_dict in vars(self).items():
             if isinstance(crops_dict, dict):
                 print(f"Saving crops from: {crop_dict_name}")
                 for idx, crop in crops_dict.items():
                     if crop is None:
                         continue
                     # שם הקובץ יכול לכלול את האינדקס המקורי ואת סוג החיתוך
                     out_name = os.path.join(out_dir, f"{idx}_{crop_dict_name}_crop.jpg")
                     try:
                         cv2.imwrite(out_name, crop)
                         ind += 1
                     except Exception as e:
                         print(f"Error saving crop {out_name}: {e}")
        print(f"Saved {ind} crops to {out_dir}")



class PersonAndFaceResult:
     # שינוי החתימה של __init__
    def __init__(self, bboxes: List[List[int]], labels: List[str], orig_img: np.ndarray, confs: Optional[List[float]] = None):

        self.orig_img = orig_img
        self.bboxes = bboxes # רשימה של [x1, y1, x2, y2]
        self.labels = labels # רשימה של שמות 'person' או 'human face' (אחרי סינון)
        self.confs = confs if confs is not None else [1.0] * len(bboxes) # ציון ביטחון אם יש

        # הגדרת מיפוי קטגוריות פנימי (יכול להיות דינמי יותר אם צריך)
        self.cat_names = {0: "person", 1: "human face"}
        self.cat_ids = {v: k for k, v in self.cat_names.items()} # הפוך

        # מציאת האינדקסים של אנשים ופנים ברשימות המסוננות
        self._person_indices = [i for i, lbl in enumerate(self.labels) if lbl == "person"]
        self._face_indices = [i for i, lbl in enumerate(self.labels) if lbl == "human face"]

        # אתחול מבני נתונים לשיוך ותוצאות
        self.face_to_person_map: Dict[int, Optional[int]] = {face_idx: None for face_idx in self._face_indices}
        # אתחול כל האנשים כלא משויכים בהתחלה
        self.unassigned_persons_inds: List[int] = list(self._person_indices)

        n_objects = len(self.bboxes)
        self.ages: List[Optional[float]] = [None] * n_objects
        self.genders: List[Optional[str]] = [None] * n_objects
        self.gender_scores: List[Optional[float]] = [None] * n_objects

        # אופציונלי: שמירת ID מעקב אם רלוונטי בעתיד
        self.track_ids: List[Optional[int]] = [None] * n_objects

        # מייד אחרי האתחול, ננסה לשייך פנים לאנשים
        self.associate_faces_with_persons()


    @property
    def n_objects(self) -> int:
        # מספר האובייקטים הכולל (אנשים + פנים) שהועברו
        return len(self.bboxes)

    @property
    def n_faces(self) -> int:
        return len(self._face_indices)

    @property
    def n_persons(self) -> int:
        return len(self._person_indices)

    def get_bboxes_inds(self, category: str) -> List[int]:
        """ מחזיר את האינדקסים ברשימות הפנימיות עבור קטגוריה נתונה """
        if category == "person":
            return list(self._person_indices)
        elif category == "human face":
            return list(self._face_indices)
        else:
            # אפשר להרחיב לקטגוריות אחרות אם Florence זיהה אותן והעברנו אותן
            return [i for i, lbl in enumerate(self.labels) if lbl == category]

    def get_distance_to_center(self, bbox_ind: int) -> float:
        """ מחשב מרחק אוקלידי ממרכז התיבה למרכז התמונה """
        if bbox_ind >= len(self.bboxes):
             return float('inf') # אינדקס לא חוקי
        im_h, im_w = self.orig_img.shape[:2]
        x1, y1, x2, y2 = self.bboxes[bbox_ind]
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        dist = math.dist([center_x, center_y], [im_w / 2, im_h / 2])
        return dist

    def plot(
        self,
        conf_scores=False, # האם להציג ציון ביטחון (אם קיים)
        line_width=None,
        font_size=None,
        font="Arial.ttf", # ודא שהפונט קיים או השתמש בברירת מחדל
        pil=False,
        img_to_plot_on=None, # תמונה לצייר עליה (אם לא המקורית)
        labels=True,
        boxes=True,
        ages=True,
        genders=True,
        gender_probs=False,
    ):
        """ מצייר את תוצאות הזיהוי והערכת הגיל/מגדר על התמונה """
        if img_to_plot_on is None:
            img_to_plot_on = self.orig_img
        # שימוש ב-Annotator של ultralytics עדיין אפשרי ונוח
        annotator = Annotator(
            deepcopy(img_to_plot_on),
            line_width,
            font_size,
            font,
            pil,
            example=list(self.cat_names.values()), # ניתן לו את שמות הקטגוריות שלנו
        )

        if boxes and len(self.bboxes) > 0:
            # קביעת צבעים לפי שיוך פנים-אדם
            colors_by_ind = {}
            color_counter = 2 # התחלת צבעים מותאמים
            for face_ind, person_ind in self.face_to_person_map.items():
                if person_ind is not None:
                    # אם הפנים משוייכות לאדם, שניהם יקבלו אותו צבע
                    current_color = color_counter
                    colors_by_ind[face_ind] = current_color
                    colors_by_ind[person_ind] = current_color
                    color_counter += 1
                else:
                    # פנים לא משוייכות - צבע ברירת מחדל 1
                     colors_by_ind[face_ind] = 1 # למשל כחול
            for person_ind in self.unassigned_persons_inds:
                 # אדם לא משוייך - צבע ברירת מחדל 0
                 colors_by_ind[person_ind] = 0 # למשל אדום

            # מעבר על כל התיבות שהועברו (אנשים ופנים)
            for bb_ind, (bbox, label_str, conf_val, age, gender, gender_score) in enumerate(
                zip(self.bboxes, self.labels, self.confs, self.ages, self.genders, self.gender_scores)
            ):
                # בניית התווית להצגה
                display_label = ""
                if labels:
                     display_label += label_str # שם הקטגוריה ('person' או 'human face')
                if conf_scores and conf_val is not None:
                     display_label += f" {conf_val:.2f}"
                if ages and age is not None:
                    display_label += f" Age: {age:.1f}"
                if genders and gender is not None:
                    display_label += f" {'F' if gender == 'female' else 'M'}"
                if gender_probs and gender_score is not None:
                    display_label += f" ({gender_score*100:.0f}%)" # הצגת ציון מגדר באחוזים

                # קבלת הצבע המתאים
                plot_color = colors(colors_by_ind.get(bb_ind, 0), True) # ברירת מחדל צבע 0 אם האינדקס לא נמצא

                # ציור התיבה והתווית
                annotator.box_label(bbox, display_label.strip(), color=plot_color)

        return annotator.result()


    def _get_id_by_ind(self, ind: Optional[int] = None) -> int:
         """ מחזיר ID מעקב אם קיים (כרגע לא בשימוש עם Florence-2) """
         if ind is None or ind >= len(self.track_ids) or self.track_ids[ind] is None:
             return -1
         return self.track_ids[ind]

    def get_bbox_by_ind(self, ind: int, clamp: bool = True) -> Optional[torch.tensor]:
         """ מחזיר את התיבה התוחמת באינדקס הנתון כ-tensor """
         if ind >= len(self.bboxes):
             return None
         bb = torch.tensor(self.bboxes[ind], dtype=torch.int32) # המרה ל-tensor
         if clamp:
             im_h, im_w = self.orig_img.shape[:2]
             bb[0] = torch.clamp(bb[0], min=0, max=im_w - 1) # x1
             bb[1] = torch.clamp(bb[1], min=0, max=im_h - 1) # y1
             bb[2] = torch.clamp(bb[2], min=0, max=im_w - 1) # x2
             bb[3] = torch.clamp(bb[3], min=0, max=im_h - 1) # y2
         return bb


    def set_age(self, ind: Optional[int], age: float):
        if ind is not None and ind < len(self.ages):
            self.ages[ind] = age

    def set_gender(self, ind: Optional[int], gender: str, gender_score: float):
        if ind is not None and ind < len(self.genders):
            self.genders[ind] = gender
            self.gender_scores[ind] = gender_score

    @staticmethod
    def _gather_tracking_result(
        tracked_objects: Dict[int, List[AGE_GENDER_INFO]], # שינוי טיפוס
        fguid: int = -1,
        pguid: int = -1,
        minimum_sample_size: int = 10,
    ) -> AGE_GENDER_INFO:
        """ אגרגציה של תוצאות מעקב (דורש עדכון אם משתמשים במעקב) """
        # פונקציה זו מניחה שקיים מנגנון מעקב נפרד שמספק tracked_objects
        # כרגע לא רלוונטי ל-Florence-2 הבסיסי

        assert fguid != -1 or pguid != -1, "Incorrect tracking behaviour"

        # חילוץ נתונים מההיסטוריה
        def extract_data(guid, index):
             if guid in tracked_objects:
                 return [r[index] for r in tracked_objects[guid] if r is not None and r[index] is not None]
             return []

        face_ages = extract_data(fguid, 0)
        face_genders = extract_data(fguid, 1)
        # face_gender_scores = extract_data(fguid, 2) # אם רוצים לשקלל לפי ציון

        person_ages = extract_data(pguid, 0)
        person_genders = extract_data(pguid, 1)
        # person_gender_scores = extract_data(pguid, 2)

        final_age: Optional[float] = None
        final_gender: Optional[str] = None
        final_gender_score: Optional[float] = None # ממוצע ציונים או אחר

        all_ages = face_ages + person_ages
        if all_ages:
             if len(all_ages) >= minimum_sample_size:
                 final_age = aggregate_votes_winsorized(np.array(all_ages))
             else:
                 final_age = np.mean(all_ages)

        all_genders = face_genders + person_genders
        if all_genders:
            # מציאת המגדר השכיח ביותר
            from collections import Counter
            gender_counts = Counter(all_genders)
            if gender_counts:
                 final_gender = gender_counts.most_common(1)[0][0]
                 # חישוב ציון ממוצע עבור המגדר שנבחר (אופציונלי)
                 # related_scores = [gs for g, gs in zip(face_genders + person_genders, face_gender_scores + person_gender_scores) if g == final_gender and gs is not None]
                 # final_gender_score = np.mean(related_scores) if related_scores else 1.0

        return final_age, final_gender, final_gender_score


    def get_results_for_tracking(self) -> Tuple[Dict[int, AGE_GENDER_INFO], Dict[int, AGE_GENDER_INFO]]:
        """ מחזיר את תוצאות הגיל/מגדר עבור אובייקטים עם ID מעקב (אם קיים) """
        persons_tracked: Dict[int, AGE_GENDER_INFO] = {}
        faces_tracked: Dict[int, AGE_GENDER_INFO] = {}

        for idx, track_id in enumerate(self.track_ids):
             if track_id is not None:
                 age_gender_data = (self.ages[idx], self.genders[idx], self.gender_scores[idx])
                 if self.labels[idx] == "person":
                     persons_tracked[track_id] = age_gender_data
                 elif self.labels[idx] == "human face":
                     faces_tracked[track_id] = age_gender_data

        return persons_tracked, faces_tracked

    def associate_faces_with_persons(self):
        """ משייך פנים לאנשים על בסיס שילוב של IoU, מרחק והכלה. """
        face_inds = self._face_indices
        person_inds = self._person_indices

        if not face_inds or not person_inds:
            self.face_to_person_map = {face_idx: None for face_idx in face_inds}
            self.unassigned_persons_inds = list(person_inds)
            return

        num_persons = len(person_inds)
        num_faces = len(face_inds)

        # Convert bboxes to tensors for IoU calculation
        face_bboxes_t = torch.stack([self.get_bbox_by_ind(ind, clamp=False) for ind in face_inds])
        person_bboxes_t = torch.stack([self.get_bbox_by_ind(ind, clamp=False) for ind in person_inds])

        # 1. Calculate base IoU (or IoU over second box for potential containment)
        # Using IoU over the face box might be better here: IoU(person, face, over_second=True)
        iou_matrix = box_iou(person_bboxes_t, face_bboxes_t, over_second=True).cpu().numpy() # IoU = Inter / Area(Face)

        # 2. Calculate a "cost" or "score" matrix incorporating other factors
        # We want to maximize the score, so higher is better.
        # Start with IoU as the base score.
        score_matrix = np.zeros((num_persons, num_faces))

        # בונוס על הכלה
        containment_bonus = 0.2 # בונוס קטן אם הפנים מוכלות בגוף

        # קנס על מרחק
        distance_penalty_factor = 0.5 # מקדם השפעה של קנס המרחק

        for p_idx_in_list, original_p_idx in enumerate(person_inds):
             person_bbox = person_bboxes_t[p_idx_in_list].tolist()
             person_height = person_bbox[3] - person_bbox[1]
             for f_idx_in_list, original_f_idx in enumerate(face_inds):
                 face_bbox = face_bboxes_t[f_idx_in_list].tolist()
                 base_iou_score = iou_matrix[p_idx_in_list, f_idx_in_list]

                 # Calculate distance penalty
                 dist_penalty = calculate_distance_penalty(face_bbox, person_bbox, person_height)

                 # Calculate containment bonus
                 bonus = containment_bonus if is_contained(face_bbox, person_bbox) else 0

                 # Combine scores: IoU - distance_penalty + containment_bonus
                 # Ensure score stays non-negative
                 combined_score = max(0, base_iou_score - (dist_penalty * distance_penalty_factor) + bonus)

                 # אפשר להוסיף תנאי סף - אם ה-IoU נמוך מאוד, אולי לא כדאי לשקול בכלל
                 # if base_iou_score < 0.01:
                 #     combined_score = -1 # ציון שלילי כדי למנוע התאמה

                 score_matrix[p_idx_in_list, f_idx_in_list] = combined_score


        # 3. Use Hungarian algorithm on the combined score matrix (maximizing score)
        person_indices_in_list, face_indices_in_list = linear_sum_assignment(score_matrix, maximize=True)

        # 4. Process assignments based on a minimum score threshold
        MIN_ASSIGNMENT_SCORE = 0.1 # סף מינימלי לשיוך (ניתן לכוונון)

        new_face_to_person_map = {face_idx: None for face_idx in face_inds} # אתחול מחדש
        assigned_persons = set()

        for p_idx_in_list, f_idx_in_list in zip(person_indices_in_list, face_indices_in_list):
            score = score_matrix[p_idx_in_list, f_idx_in_list]
            original_face_idx = face_inds[f_idx_in_list]
            original_person_idx = person_inds[p_idx_in_list]

            # שייך רק אם הציון מספיק גבוה והאדם לא שויך כבר
            if score >= MIN_ASSIGNMENT_SCORE and original_person_idx not in assigned_persons:
                 new_face_to_person_map[original_face_idx] = original_person_idx
                 assigned_persons.add(original_person_idx)

        self.face_to_person_map = new_face_to_person_map
        self.unassigned_persons_inds = [p_idx for p_idx in person_inds if p_idx not in assigned_persons]



    def crop_object(
        self, full_image: np.ndarray, ind: int, cut_other_classes: Optional[List[str]] = None, expand_ratio: float = 0.0
    ) -> Optional[np.ndarray]:
        """
        חותך אובייקט מהתמונה המלאה, עם אפשרות לחתוך אובייקטים אחרים החופפים
        והגדלה קלה של התיבה.
        """
        if ind >= len(self.bboxes):
            return None

        im_h, im_w = full_image.shape[:2]
        obj_bbox = self.get_bbox_by_ind(ind, clamp=False) # קבלת התיבה המקורית
        if obj_bbox is None: return None

        x1, y1, x2, y2 = obj_bbox.tolist() # המרה לרשימה

        # הרחבת התיבה התוחמת (אופציונלי)
        if expand_ratio > 0:
             w = x2 - x1
             h = y2 - y1
             expand_w = w * expand_ratio / 2
             expand_h = h * expand_ratio / 2
             x1 = int(max(0, x1 - expand_w))
             y1 = int(max(0, y1 - expand_h))
             x2 = int(min(im_w, x2 + expand_w))
             y2 = int(min(im_h, y2 + expand_h))

        # חישוב הגבולות החתוכים הסופיים (בתוך גבולות התמונה)
        crop_x1 = max(0, x1)
        crop_y1 = max(0, y1)
        crop_x2 = min(im_w, x2)
        crop_y2 = min(im_h, y2)

        # חיתוך ראשוני
        obj_image = full_image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        if obj_image.size == 0: return None # במקרה שהתיבה מחוץ לתמונה לגמרי
        crop_h, crop_w = obj_image.shape[:2]

        # בדיקת גודל מינימלי (רלוונטי בעיקר לאנשים)
        MIN_PERSON_SIZE = 30 # גודל מינימלי בפיקסלים (ניתן לכוונון)
        cur_cat = self.labels[ind]
        if cur_cat == "person" and (crop_h < MIN_PERSON_SIZE or crop_w < MIN_PERSON_SIZE):
             print(f"Skipping small person crop: ({crop_w}x{crop_h}) at index {ind}")
             return None

        # אם לא צריך לחתוך אובייקטים אחרים, נחזיר את החיתוך
        if not cut_other_classes:
            return obj_image

        # --- לוגיקת חיתוך אובייקטים חופפים ---
        IOU_THRESH = 0.00001 # סף חפיפה מינימלי
        MIN_PERSON_CROP_AFTERCUT_RATIO = 0.3 # יחס מינימלי של פיקסלים לא שחורים אחרי חיתוך
        CROP_ROUND_RATE = 0.1 # עד כמה קרוב לקצה התיבה לחתוך (רלוונטי בעיקר לחיתוך גופות)

        # בניית תיבה מורחבת כטנסור להשוואה
        expanded_bbox_tensor = torch.tensor([x1, y1, x2, y2], dtype=torch.int32)

        # יצירת רשימת כל התיבות האחרות כטנסורים
        other_bboxes_t = []
        other_indices = []
        for other_ind in range(len(self.bboxes)):
             if ind == other_ind: continue # לא להשוות לעצמו
             other_bb = self.get_bbox_by_ind(other_ind, clamp=True) # קבל כטנסור חתוך לגבולות
             if other_bb is not None:
                 other_bboxes_t.append(other_bb)
                 other_indices.append(other_ind)

        if not other_bboxes_t: # אם אין אובייקטים אחרים
             return obj_image

        # חישוב IoU בין התיבה המורחבת לכל האחרות
        iou_matrix = box_iou(expanded_bbox_tensor.unsqueeze(0), torch.stack(other_bboxes_t)).squeeze(0).cpu().numpy()


        # חיתוך (השחרה) של אזורים חופפים מאובייקטים אחרים
        for i, other_ind in enumerate(other_indices):
            iou = iou_matrix[i]
            other_cat = self.labels[other_ind]

            if iou < IOU_THRESH or other_cat not in cut_other_classes:
                continue

            # קואורדינטות התיבה האחרת (מוגבלות לגבולות התמונה)
            o_x1_orig, o_y1_orig, o_x2_orig, o_y2_orig = other_bboxes_t[i].tolist()

            # מיפוי קואורדינטות התיבה האחרת לקואורדינטות של התמונה החתוכה (obj_image)
            # יש לקחת בחשבון את ההזזה (offset) שנגרמה מהחיתוך הראשוני (crop_x1, crop_y1)
            o_x1_crop = max(0, o_x1_orig - crop_x1)
            o_y1_crop = max(0, o_y1_orig - crop_y1)
            o_x2_crop = min(crop_w, o_x2_orig - crop_x1)
            o_y2_crop = min(crop_h, o_y2_orig - crop_y1)

            # החלת עיגול פינות אם האובייקט הנחתך אינו פנים
            if other_cat != "human face":
                h_crop_relative = o_y1_crop / crop_h if crop_h > 0 else 0
                h_crop_relative_end = (crop_h - o_y2_crop) / crop_h if crop_h > 0 else 0
                w_crop_relative = o_x1_crop / crop_w if crop_w > 0 else 0
                w_crop_relative_end = (crop_w - o_x2_crop) / crop_w if crop_w > 0 else 0

                if h_crop_relative < CROP_ROUND_RATE: o_y1_crop = 0
                if h_crop_relative_end < CROP_ROUND_RATE: o_y2_crop = crop_h
                if w_crop_relative < CROP_ROUND_RATE: o_x1_crop = 0
                if w_crop_relative_end < CROP_ROUND_RATE: o_x2_crop = crop_w

            # ביצוע ההשחרה
            if o_x1_crop < o_x2_crop and o_y1_crop < o_y2_crop: # ודא שיש אזור לחתוך
                 obj_image[o_y1_crop:o_y2_crop, o_x1_crop:o_x2_crop] = 0

        # בדיקת כמות הפיקסלים שנותרו
        if cur_cat == "person": # בדיקה זו רלוונטית בעיקר לאנשים
            if obj_image.ndim == 3 and obj_image.shape[2] > 0: # בדוק אם התמונה לא ריקה ויש לה ערוצים
                non_zero_pixels = np.count_nonzero(obj_image)
                total_pixels = obj_image.shape[0] * obj_image.shape[1] * obj_image.shape[2]
                if total_pixels > 0:
                    remain_ratio = non_zero_pixels / total_pixels
                    if remain_ratio < MIN_PERSON_CROP_AFTERCUT_RATIO:
                        print(f"Skipping person crop index {ind} due to low remaining ratio after cutting: {remain_ratio:.2f}")
                        return None
                else:
                    return None # תמונה ריקה
            else: # אם התמונה הפכה לריקה או לא תקינה
                return None

        return obj_image


    def collect_crops(self, image, expand_ratio_person=0.1, expand_ratio_face=0.1) -> PersonAndFaceCrops:
        """
        אוסף את החיתוכים הנדרשים (פנים, גופות) מהתמונה המלאה.
        משתמש במיפוי פנים-אדם שחושב קודם.
        """
        crops_data = PersonAndFaceCrops()

        # מעבר על הפנים המשוייכות לאנשים
        for face_ind, person_ind in self.face_to_person_map.items():
             # חיתוך פנים (בלי לחתוך אובייקטים אחרים מהפנים עצמן, אולי עם הרחבה קלה)
             face_image = self.crop_object(image, face_ind, cut_other_classes=[], expand_ratio=expand_ratio_face)
             if face_image is None: continue # דילוג אם החיתוך נכשל

             if person_ind is None:
                 # פנים ללא גוף משויך
                 crops_data.crops_faces_wo_body[face_ind] = face_image
             else:
                 # פנים עם גוף משויך
                 # חיתוך גוף - עם חיתוך פנים ואנשים אחרים, ועם הרחבה
                 person_image = self.crop_object(image, person_ind, cut_other_classes=["human face", "person"], expand_ratio=expand_ratio_person)

                 if person_image is not None:
                      # רק אם גם חיתוך הגוף הצליח
                      crops_data.crops_faces[face_ind] = face_image
                      crops_data.crops_persons[person_ind] = person_image
                 else:
                      # אם חיתוך הגוף נכשל, נתייחס לפנים כאילו אין להן גוף משויך
                      crops_data.crops_faces_wo_body[face_ind] = face_image


        # מעבר על האנשים שלא שויכו לפנים
        for person_ind in self.unassigned_persons_inds:
            # חיתוך גוף - עם חיתוך פנים ואנשים אחרים, ועם הרחבה
            person_image = self.crop_object(image, person_ind, cut_other_classes=["human face", "person"], expand_ratio=expand_ratio_person)
            if person_image is not None:
                crops_data.crops_persons_wo_face[person_ind] = person_image

        # אופציונלי: הדפסה או שמירה של החיתוכים לצורך דיבוג
        # print(f"Collected {len(crops_data.crops_faces)} faces with body.")
        # print(f"Collected {len(crops_data.crops_faces_wo_body)} faces without body.")
        # print(f"Collected {len(crops_data.crops_persons_wo_face)} persons without face.")
        # crops_data.save(out_dir="debug_crops")
        return crops_data