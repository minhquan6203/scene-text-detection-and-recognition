import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import re
import torch
from collections import OrderedDict
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO

from detectron2.utils import comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator

import glob
import shutil
from shapely.geometry import Polygon, LinearRing
from detectron2.evaluation import text_eval_script
from detectron2.evaluation import text_eval_script_ic15
import zipfile
import pickle
import cv2
import editdistance
class TextEvaluator(DatasetEvaluator):
    """
    Evaluate text proposals and recognition.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        self._tasks = ("polygon", "recognition")
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            raise AttributeError(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'."
            )
        
        CTLABELS = [" ","!",'"',"#","$","%","&","'","(",")","*","+",",","-",".","/","0","1","2","3","4","5","6","7","8","9",":",";","<","=",">","?","@","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","[","\\","]","^","_","`","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","{","|","}","~","ˋ","ˊ","﹒","ˀ","˜","ˇ","ˆ","˒","‑",'´', "~"]

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        self.dataset_name = dataset_name
        # use dataset_name to decide eval_gt_path
        self.lexicon_type = 3
        if "totaltext" in dataset_name:
            self._text_eval_gt_path = "datasets/evaluation/gt_totaltext.zip"
            self._word_spotting = True
            self.dataset_name = "totaltext"
        elif "ctw1500" in dataset_name:
            self._text_eval_gt_path = "datasets/evaluation/gt_ctw1500.zip"
            self._word_spotting = False
            self.dataset_name = "ctw1500"
        elif "icdar2015" in dataset_name:
            self._text_eval_gt_path = "datasets/evaluation/gt_icdar2015.zip"
            self._word_spotting = False
            self.dataset_name = "icdar2015"
        elif "vintext" in dataset_name:
            self.lexicon_type = None
            self._text_eval_gt_path = "datasets/evaluation/gt_vintext.zip"
            self._word_spotting = True
        elif "custom" in dataset_name:
            self._text_eval_gt_path = "datasets/evaluation/gt_custom.zip"
            self._word_spotting = False
        self._text_eval_confidence = cfg.TEST.INFERENCE_TH_TEST
        self.nms_enable = cfg.TEST.USE_NMS_IN_TSET

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            instances = output["instances"].to(self._cpu_device)
            prediction["instances"] = self.instances_to_coco_json(instances, input)
            self._predictions.append(prediction)

    def to_eval_format(self, file_path, temp_dir="temp_det_results", cf_th=0.5):
        def fis_ascii(s):
            a = (ord(c) < 128 for c in s)
            return all(a)

        def de_ascii(s):
            a = [c for c in s if ord(c) < 128]
            outa = ''
            for i in a:
                outa +=i
            return outa

        with open(file_path, 'r') as f:
            data = json.load(f)
            with open('temp_all_det_cors.txt', 'w') as f2:
                for ix in range(len(data)):
                    if data[ix]['score'] > 0.1:
                        outstr = '{}: '.format(data[ix]['image_id'])
                        xmin = 1000000
                        ymin = 1000000
                        xmax = 0 
                        ymax = 0
                        for i in range(len(data[ix]['polys'])):
                            outstr = outstr + str(int(data[ix]['polys'][i][0])) +','+str(int(data[ix]['polys'][i][1])) +','
                        if not "vintext" in self.dataset_name:
                            ass = de_ascii(data[ix]['rec'])
                        else:
                            ass = data[ix]['rec']
                        if len(ass)>=0: # 
                            outstr = outstr + str(round(data[ix]['score'], 3)) +',####'+ass+'\n'	
                            f2.writelines(outstr)
                f2.close()
        dirn = temp_dir
        lsc = [cf_th] 
        fres = open('temp_all_det_cors.txt', 'r').readlines()
        for isc in lsc:	
            if not os.path.isdir(dirn):
                os.mkdir(dirn)

            for line in fres:
                line = line.strip()
                s = line.split(': ')
                filename = '{:07d}.txt'.format(int(s[0]))
                outName = os.path.join(dirn, filename)
                with open(outName, 'a') as fout:
                    ptr = s[1].strip().split(',####')
                    score = ptr[0].split(',')[-1]
                    if float(score) < isc:
                        continue
                    cors = ','.join(e for e in ptr[0].split(',')[:-1])
                    fout.writelines(cors+',####'+ptr[1]+'\n')
        os.remove("temp_all_det_cors.txt")

    def sort_detection(self, temp_dir):
        origin_file = temp_dir
        output_file = "final_"+temp_dir
        output_file_full = "full_final_"+temp_dir
        if not os.path.isdir(output_file_full):
            os.mkdir(output_file_full)
        if not os.path.isdir(output_file):
            os.mkdir(output_file)
        files = glob.glob(origin_file+'*.txt')
        files.sort()
        if "totaltext" in self.dataset_name:
            if not self.lexicon_type == None:
                lexicon_path = 'datasets/totaltext/weak_voc_new.txt'
                lexicon_fid=open(lexicon_path, 'r')
                pair_list = open('datasets/totaltext/weak_voc_pair_list.txt', 'r')
                pairs = dict()
                for line in pair_list.readlines():
                    line=line.strip()
                    word = line.split(' ')[0].upper()
                    word_gt = line[len(word)+1:]
                    pairs[word] = word_gt
                lexicon_fid=open(lexicon_path, 'r')
                lexicon=[]
                for line in lexicon_fid.readlines():
                    line=line.strip()
                    lexicon.append(line)
        elif "ctw1500" in self.dataset_name:
            if not self.lexicon_type == None:
                lexicon_path = 'datasets/CTW1500/weak_voc_new.txt'
                lexicon_fid=open(lexicon_path, 'r')
                pair_list = open('datasets/CTW1500/weak_voc_pair_list.txt', 'r')
                pairs = dict()
                lexicon_fid=open(lexicon_path, 'r')
                lexicon=[]
                for line in lexicon_fid.readlines():
                    line=line.strip()
                    lexicon.append(line)
                    pairs[line.upper()] = line
        elif "icdar2015" in self.dataset_name:
            if self.lexicon_type==1: 
                # generic lexicon
                lexicon_path = 'datasets/icdar2015/GenericVocabulary_new.txt'
                lexicon_fid=open(lexicon_path, 'r')
                pair_list = open('datasets/icdar2015/GenericVocabulary_pair_list.txt', 'r')
                pairs = dict()
                for line in pair_list.readlines():
                    line=line.strip()
                    word = line.split(' ')[0].upper()
                    word_gt = line[len(word)+1:]
                    pairs[word] = word_gt
                lexicon_fid=open(lexicon_path, 'r')
                lexicon=[]
                for line in lexicon_fid.readlines():
                    line=line.strip()
                    lexicon.append(line)
            if self.lexicon_type==2:
                # weak lexicon
                lexicon_path = 'datasets/icdar2015/ch4_test_vocabulary_new.txt'
                lexicon_fid=open(lexicon_path, 'r')
                pair_list = open('datasets/icdar2015/ch4_test_vocabulary_pair_list.txt', 'r')
                pairs = dict()
                for line in pair_list.readlines():
                    line=line.strip()
                    word = line.split(' ')[0].upper()
                    word_gt = line[len(word)+1:]
                    pairs[word] = word_gt
                lexicon_fid=open(lexicon_path, 'r')
                lexicon=[]
                for line in lexicon_fid.readlines():
                    line=line.strip()
                    lexicon.append(line)

        def find_match_word(rec_str, pairs, lexicon=None):
            rec_str = rec_str.upper()
            dist_min = 100
            dist_min_pre = 100
            match_word = ''
            match_dist = 100
            for word in lexicon:
                word = word.upper()
                ed = editdistance.eval(rec_str, word)
                length_dist = abs(len(word) - len(rec_str))
                dist = ed
                if dist<dist_min:
                    dist_min = dist
                    match_word = pairs[word]
                    match_dist = dist
            return match_word, match_dist
        for i in files:
            if "icdar2015" in self.dataset_name:
                out = output_file + 'res_img_' + str(int(i.split('/')[-1].split('.')[0])) + '.txt'
                out_full = output_file_full + 'res_img_' + str(int(i.split('/')[-1].split('.')[0])) + '.txt'
                if self.lexicon_type==3:
                    lexicon_path = 'datasets/icdar2015/new_strong_lexicon/new_voc_img_' + str(int(i.split('/')[-1].split('.')[0])) + '.txt'
                    lexicon_fid=open(lexicon_path, 'r')
                    pair_list = open('datasets/icdar2015/new_strong_lexicon/pair_voc_img_' + str(int(i.split('/')[-1].split('.')[0])) + '.txt')
                    pairs = dict()
                    for line in pair_list.readlines():
                        line=line.strip()
                        word = line.split(' ')[0].upper()
                        word_gt = line[len(word)+1:]
                        pairs[word] = word_gt
                    lexicon_fid=open(lexicon_path, 'r')
                    lexicon=[]
                    for line in lexicon_fid.readlines():
                        line=line.strip()
                        lexicon.append(line)
            else:
                out = i.replace(origin_file, output_file)
                out_full = i.replace(origin_file, output_file_full)
            fin = open(i, 'r').readlines()
            fout = open(out, 'w')
            fout_full = open(out_full, 'w')
            for iline, line in enumerate(fin):
                ptr = line.strip().split(',####')
                rec  = ptr[1]
                cors = ptr[0].split(',')
                assert(len(cors) %2 == 0), 'cors invalid.'
                pts = [(int(cors[j]), int(cors[j+1])) for j in range(0,len(cors),2)]
                try:
                    pgt = Polygon(pts)
                except Exception as e:
                    print(e)
                    print('An invalid detection in {} line {} is removed ... '.format(i, iline))
                    continue
                
                if not pgt.is_valid:
                    print('An invalid detection in {} line {} is removed ... '.format(i, iline))
                    continue
                    
                pRing = LinearRing(pts)
                if not "icdar2015" in self.dataset_name:
                    if pRing.is_ccw:
                        pts.reverse()
                outstr = ''
                for ipt in pts[:-1]:
                    outstr += (str(int(ipt[0]))+','+ str(int(ipt[1]))+',')
                outstr += (str(int(pts[-1][0]))+','+ str(int(pts[-1][1])))
                pts = outstr
                if "icdar2015" in self.dataset_name:
                    outstr = outstr + ',' + rec
                else:
                    outstr = outstr + ',####' + rec
                fout.writelines(outstr+'\n')
                if self.lexicon_type is None:
                    rec_full = rec
                else:
                    match_word, match_dist = find_match_word(rec,pairs,lexicon)
                    if match_dist<1.5:
                        rec_full = match_word
                        if "icdar2015" in self.dataset_name:
                            pts = pts + ',' + rec_full
                        else:
                            pts = pts + ',####' + rec_full
                        fout_full.writelines(pts+'\n')
            fout.close()
            fout_full.close()
        def zipdir(path, ziph):
            # ziph is zipfile handle
            for root, dirs, files in os.walk(path):
                for file in files:
                    ziph.write(os.path.join(root, file))
        if "icdar2015" in self.dataset_name:
            os.system('zip -r -q -j '+'det.zip'+' '+output_file+'/*')
            os.system('zip -r -q -j '+'det_full.zip'+' '+output_file_full+'/*')
            shutil.rmtree(origin_file)
            shutil.rmtree(output_file)
            shutil.rmtree(output_file_full)
            return "det.zip", "det_full.zip"
        else:
            os.chdir(output_file)
            zipf = zipfile.ZipFile('../det.zip', 'w', zipfile.ZIP_DEFLATED)
            zipdir('./', zipf)
            zipf.close()
            os.chdir("../")

            os.chdir(output_file_full)
            zipf_full = zipfile.ZipFile('../det_full.zip', 'w', zipfile.ZIP_DEFLATED)
            zipdir('./', zipf_full)
            zipf_full.close()
            os.chdir("../")
            # clean temp files
            shutil.rmtree(origin_file)
            shutil.rmtree(output_file)
            shutil.rmtree(output_file_full)
            return "det.zip", "det_full.zip"
    
    def evaluate_with_official_code(self, result_path, gt_path):
        if "icdar2015" in self.dataset_name:
            return text_eval_script_ic15.text_eval_main_ic15(det_file=result_path, gt_file=gt_path, is_word_spotting=self._word_spotting)
        else:
            return text_eval_script.text_eval_main(det_file=result_path, gt_file=gt_path, is_word_spotting=self._word_spotting)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        PathManager.mkdirs(self._output_dir)
        file_path = os.path.join(self._output_dir, "text_results.json")
        self._logger.info("Saving results to {}".format(file_path))
        with PathManager.open(file_path, "w") as f:
            f.write(json.dumps(coco_results))
            f.flush()
        self._results = OrderedDict()
        # eval text
        if not self._text_eval_gt_path:
            return copy.deepcopy(self._results)
        temp_dir = "temp_det_results/"
        self.to_eval_format(file_path, temp_dir, self._text_eval_confidence)
        result_path, result_path_full = self.sort_detection(temp_dir)
        text_result = self.evaluate_with_official_code(result_path, self._text_eval_gt_path) # None 
        text_result["e2e_method"] = "None-" + text_result["e2e_method"]
        if not self.lexicon_type == None:
            dict_lexicon = {"1": "Generic", "2": "Weak", "3": "Strong"}
            text_result_full = self.evaluate_with_official_code(result_path_full, self._text_eval_gt_path) # with lexicon
            text_result_full["e2e_method"] = dict_lexicon[str(self.lexicon_type)] + "-" + text_result_full["e2e_method"]
        os.remove(result_path)
        os.remove(result_path_full)
        # parse
        template = "(\S+): (\S+): (\S+), (\S+): (\S+), (\S+): (\S+)"
        result = text_result["det_only_method"]
        groups = re.match(template, result).groups()
        self._results[groups[0]] = {groups[i*2+1]: float(groups[(i+1)*2]) for i in range(3)}
        result = text_result["e2e_method"]
        groups = re.match(template, result).groups()
        self._results[groups[0]] = {groups[i*2+1]: float(groups[(i+1)*2]) for i in range(3)}
        if not self.lexicon_type == None:
            result = text_result_full["e2e_method"]
            groups = re.match(template, result).groups()
            self._results[groups[0]] = {groups[i*2+1]: float(groups[(i+1)*2]) for i in range(3)}

        return copy.deepcopy(self._results)


    def instances_to_coco_json(self, instances, inputs):
        img_id = inputs["image_id"]
        width = inputs['width']
        height = inputs['height']
        num_instances = len(instances)
        if num_instances == 0:
            return []
        scores = instances.scores.tolist()
        masks = np.asarray(instances.pred_masks)
        masks = [GenericMask(x, height, width) for x in masks]
        recs = instances.pred_rec.numpy()

        if self.nms_enable:
            polys = []
            for mask in masks:
                if not len(mask.polygons):
                    continue
                polys.append(np.concatenate(mask.polygons).reshape(-1,2))
            keep = self.py_cpu_pnms(polys,scores,0.5)

        results = []
        i = 0
        for mask, rec, score in zip(masks, recs, scores):
            if not len(mask.polygons):
                continue
            if self.nms_enable:
                if i not in keep:
                    i = i+1
                    continue
            poly = polys[i]
            if 'icdar2015'  in self.dataset_name:
                poly = polygon2rbox(poly, height, width)
                poly = np.array(poly)
            rec_string = self.decode(rec)
            if not len(rec_string):
                i = i+1
                continue
            result = {
                "image_id": img_id,
                "category_id": 1,
                "polys": poly.tolist(),
                "rec": rec_string,
                "score": score,
            }
            results.append(result)
            i = i+1
        return results
  
    def decode(self, rec):
        CTLABELS = [" ","!",'"',"#","$","%","&","'","(",")","*","+",",","-",".","/","0","1","2","3","4","5","6","7","8","9",":",";","<","=",">","?","@","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","[","\\","]","^","_","`","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","{","|","}","~","ˋ","ˊ","﹒","ˀ","˜","ˇ","ˆ","˒","‑",'´', "~"]
        s = ''
        for c in rec:
            c = int(c)
            if 0<c < len(CTLABELS):
                if not "ctw1500" in self.dataset_name and not "vintext" in self.dataset_name:
                    if CTLABELS[c-1] in "_0123456789abcdefghijklmnopqrstuvwxyz":
                        s += CTLABELS[c-1]
                else:
                    s += CTLABELS[c-1]
            else:
                s += u''
        if "vintext" in self.dataset_name:
            s = vintext_decoder(s)
        return s

    def py_cpu_pnms(self, dets, scores, thresh):
        pts = dets
        scores = np.array(scores)
        order = scores.argsort()[::-1]
        areas = np.zeros(scores.shape)
        order = scores.argsort()[::-1]
        inter_areas = np.zeros((scores.shape[0], scores.shape[0]))
        for il in range(len(pts)):
            poly = Polygon(pts[il]).buffer(0.001)
            areas[il] = poly.area
            for jl in range(il, len(pts)):
                polyj = Polygon(pts[jl].tolist()).buffer(0.001)
                inS = poly.intersection(polyj)
                try:
                    inter_areas[il][jl] = inS.area
                except:
                    import pdb;pdb.set_trace()
                inter_areas[jl][il] = inS.area

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            ovr = inter_areas[i][order[1:]] / ((areas[i]) + areas[order[1:]] - inter_areas[i][order[1:]])
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

def polygon2rbox(polygon, image_height, image_width):
    poly = np.array(polygon).reshape((-1, 2)).astype(np.float32)
    rect = cv2.minAreaRect(poly)
    corners = cv2.boxPoints(rect)
    corners = np.array(corners, dtype="int")
    pts = get_tight_rect(corners, 0, 0, image_height, image_width, 1)
    pts = np.array(pts).reshape(-1,2)
    pts = pts.tolist()
    return pts

def get_tight_rect(points, start_x, start_y, image_height, image_width, scale):
    points = list(points)
    ps = sorted(points, key=lambda x: x[0])

    if ps[1][1] > ps[0][1]:
        px1 = ps[0][0] * scale + start_x
        py1 = ps[0][1] * scale + start_y
        px4 = ps[1][0] * scale + start_x
        py4 = ps[1][1] * scale + start_y
    else:
        px1 = ps[1][0] * scale + start_x
        py1 = ps[1][1] * scale + start_y
        px4 = ps[0][0] * scale + start_x
        py4 = ps[0][1] * scale + start_y
    if ps[3][1] > ps[2][1]:
        px2 = ps[2][0] * scale + start_x
        py2 = ps[2][1] * scale + start_y
        px3 = ps[3][0] * scale + start_x
        py3 = ps[3][1] * scale + start_y
    else:
        px2 = ps[3][0] * scale + start_x
        py2 = ps[3][1] * scale + start_y
        px3 = ps[2][0] * scale + start_x
        py3 = ps[2][1] * scale + start_y

    px1 = min(max(px1, 1), image_width - 1)
    px2 = min(max(px2, 1), image_width - 1)
    px3 = min(max(px3, 1), image_width - 1)
    px4 = min(max(px4, 1), image_width - 1)
    py1 = min(max(py1, 1), image_height - 1)
    py2 = min(max(py2, 1), image_height - 1)
    py3 = min(max(py3, 1), image_height - 1)
    py4 = min(max(py4, 1), image_height - 1)
    return [px1, py1, px2, py2, px3, py3, px4, py4]

class GenericMask:
    """
    Attribute:
        polygons (list[ndarray]): list[ndarray]: polygons for this mask.
            Each ndarray has format [x, y, x, y, ...]
        mask (ndarray): a binary mask
    """

    def __init__(self, mask_or_polygons, height, width):
        self._mask = self._polygons = self._has_holes = None
        self.height = height
        self.width = width

        m = mask_or_polygons
        if isinstance(m, dict):
            # RLEs
            assert "counts" in m and "size" in m
            if isinstance(m["counts"], list):  # uncompressed RLEs
                h, w = m["size"]
                assert h == height and w == width
                m = mask_util.frPyObjects(m, h, w)
            self._mask = mask_util.decode(m)[:, :]
            return

        if isinstance(m, list):  # list[ndarray]
            self._polygons = [np.asarray(x).reshape(-1) for x in m]
            return

        if isinstance(m, np.ndarray):  # assumed to be a binary mask
            assert m.shape[1] != 2, m.shape
            assert m.shape == (height, width), m.shape
            self._mask = m.astype("uint8")
            return

        raise ValueError("GenericMask cannot handle object {} of type '{}'".format(m, type(m)))

    @property
    def mask(self):
        if self._mask is None:
            self._mask = self.polygons_to_mask(self._polygons)
        return self._mask

    @property
    def polygons(self):
        if self._polygons is None:
            self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
        return self._polygons

    @property
    def has_holes(self):
        if self._has_holes is None:
            if self._mask is not None:
                self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
            else:
                self._has_holes = False  # if original format is polygon, does not have holes
        return self._has_holes

    def mask_to_polygons(self, mask):
        # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
        # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
        # Internal contours (holes) are placed in hierarchy-2.
        # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
        mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
        #res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return [], False
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        res = res[-2]
        res = [x.flatten() for x in res]
        # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
        # We add 0.5 to turn them into real-value coordinate space. A better solution
        # would be to first +0.5 and then dilate the returned polygon by 0.5.
        res = [x + 0.5 for x in res if len(x) >= 6]
        return res, has_holes

    def polygons_to_mask(self, polygons):
        rle = mask_util.frPyObjects(polygons, self.height, self.width)
        rle = mask_util.merge(rle)
        return mask_util.decode(rle)[:, :]

    def area(self):
        return self.mask.sum()

    def bbox(self):
        p = mask_util.frPyObjects(self.polygons, self.height, self.width)
        p = mask_util.merge(p)
        bbox = mask_util.toBbox(p)
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        return bbox

dictionary = "aàáạảãâầấậẩẫăằắặẳẵAÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪeèéẹẻẽêềếệểễEÈÉẸẺẼÊỀẾỆỂỄoòóọỏõôồốộổỗơờớợởỡOÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠiìíịỉĩIÌÍỊỈĨuùúụủũưừứựửữƯỪỨỰỬỮUÙÚỤỦŨyỳýỵỷỹYỲÝỴỶỸ"


def make_groups():
    groups = []
    i = 0
    while i < len(dictionary) - 5:
        group = [c for c in dictionary[i : i + 6]]
        i += 6
        groups.append(group)
    return groups


groups = make_groups()

TONES = ["", "ˋ", "ˊ", "﹒", "ˀ", "˜"]
SOURCES = ["ă", "â", "Ă", "Â", "ê", "Ê", "ô", "ơ", "Ô", "Ơ", "ư", "Ư", "Đ", "đ"]
TARGETS = ["aˇ", "aˆ", "Aˇ", "Aˆ", "eˆ", "Eˆ", "oˆ", "o˒", "Oˆ", "O˒", "u˒", "U˒", "D-", "d‑"]


def correct_tone_position(word):
    word = word[:-1]
    if len(word) < 2:
        pass
    first_ord_char = ""
    second_order_char = ""
    for char in word:
        for group in groups:
            if char in group:
                second_order_char = first_ord_char
                first_ord_char = group[0]
    if word[-1] == first_ord_char and second_order_char != "":
        pair_chars = ["qu", "Qu", "qU", "QU", "gi", "Gi", "gI", "GI"]
        for pair in pair_chars:
            if pair in word and second_order_char in ["u", "U", "i", "I"]:
                return first_ord_char
        return second_order_char
    return first_ord_char


def vintext_decoder(recognition):
    for char in TARGETS:
        recognition = recognition.replace(char, SOURCES[TARGETS.index(char)])
    if len(recognition) < 1:
        return recognition
    if recognition[-1] in TONES:
        if len(recognition) < 2:
            return recognition
        replace_char = correct_tone_position(recognition)
        tone = recognition[-1]
        recognition = recognition[:-1]
        for group in groups:
            if replace_char in group:
                recognition = recognition.replace(replace_char, group[TONES.index(tone)])
    return recognition
