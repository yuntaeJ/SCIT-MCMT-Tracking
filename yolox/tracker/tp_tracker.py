import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

# from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState

from collections import deque
from scipy.fftpack import fft, ifft

class LimitedQueue:
    def __init__(self, max_length=48):
        self.queue = deque(maxlen=max_length)

    def add(self, item):
        self.queue.append(item)
        
    def reset(self):
        self.queue.clear()
    
    def to_array(self):
        return np.array(self.queue)
    
    def length(self):
        return len(self.queue)
    
    def __str__(self):
        return str(self.queue)

class STrack(BaseTrack):
    # shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, occluded_val=False):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        # self.kalman_filter = None
        # self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0
        
        self.occluded = occluded_val
        
        ## parameter for TP
        self.obs_seq_len = 8
        self.pred_seq_len = 12
        self.frames_per_set = 6
        
        ## var
        self.tlwh_queue = LimitedQueue(max_length=self.obs_seq_len*self.frames_per_set)
        self.overlap_len = 0
        self.pred_traj = [[]]
        self.last_tlbr = None
        
#     def predict(self):
#         mean_state = self.mean.copy()
#         if self.state != TrackState.Tracked:
#             mean_state[7] = 0
#         self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

#     @staticmethod
#     def multi_predict(stracks):
#         if len(stracks) > 0:
#             multi_mean = np.asarray([st.mean.copy() for st in stracks])
#             multi_covariance = np.asarray([st.covariance for st in stracks])
#             for i, st in enumerate(stracks):
#                 if st.state != TrackState.Tracked:
#                     multi_mean[i][7] = 0
#             multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
#             for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
#                 stracks[i].mean = mean
#                 stracks[i].covariance = cov

    def activate(self, frame_id): 
        """Start a new tracklet"""
        # self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        # self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        
        #initally defined the tlwh when call STrack
        if not self.occluded:
            self.tlwh_queue.add(self._tlwh)
        
    def re_activate(self, new_track, frame_id, new_id=False):
        # self.mean, self.covariance = self.kalman_filter.update(
        #     self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        # )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        new_tlwh = new_track.tlwh
        self._tlwh = new_tlwh
        
        self.tlwh_queue.reset()
        if not new_track.occluded:
            self.tlwh_queue.add(self._tlwh)

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self._tlwh = new_tlwh
        # self.mean, self.covariance = self.kalman_filter.update(
        #     self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        
        if not new_track.occluded:
            self.tlwh_queue.add(self._tlwh)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        return self._tlwh.copy()

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class TPTracker(object):
    def __init__(self, tp_model, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args

        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        # self.kalman_filter = KalmanFilter()
        
        self.tp_model = tp_model
        
        ## parameter for TP
        self.obs_seq_len = 8
        self.pred_seq_len = 12
        self.frames_per_set = 6
        

    def trajectory_prediction(self, strack_pool, pred_index):

        KSTEPS = self.obs_seq_len + self.pred_seq_len
        
        obj_traj_temp = []
        for i in pred_index:
            array_representation = strack_pool[i].tlwh_queue.to_array()
            strack_pool[i].last_tlbr = array_representation[-1]
            x_coords = []
            y_coords = []
            for item in array_representation:
                x, y, w, h = item[0],item[1],item[2],item[3]
                x_coords.append(x+(w/2))
                y_coords.append(y+h)

            # FFT filter parameters
            cutoff_freq_ratio = 0.3
            averaged_coordinates = average_coordinates(np.array(x_coords), np.array(y_coords), cutoff_freq_ratio, self.frames_per_set)
            obj_traj_temp.append(averaged_coordinates)

        # create the tensor
        obs_traj = torch.tensor(obj_traj_temp).unsqueeze(0)
        V_obs = calculate_v_obs(obs_traj)
        
        obs_traj = torch.tensor(obs_traj, device='cuda:0', dtype=torch.float64)
        V_obs = torch.tensor(V_obs, device='cuda:0', dtype=torch.float64)
        
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)
        V_predx = self.tp_model(V_obs_tmp, obs_traj, KSTEPS=KSTEPS)
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy())
        
        num_of_objs = V_obs.shape[-2]
        pred_result = []
        for k in range(KSTEPS):
            V_pred = V_predx[k:k + 1, ...]
            
            V_pred = V_pred.permute(0, 2, 3, 1)
            V_pred = V_pred[0]
            final_V_pred = V_pred.data.cpu().numpy()
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(
                final_V_pred, V_x[-1, :, :].copy())

            for n in range(num_of_objs):
                pred_result.append(V_pred_rel_to_abs[:, n:n + 1, :])
        
        pred_result_final = []
        for n in range(num_of_objs):
            pred_result_final.append(pred_result[n])
        
        for i in range(len(pred_index)):
            strack_pool[pred_index[i]].pred_traj =  [coord[0] for coord in pred_result_final[i]]
                  
        return strack_pool
        
        
    def process_occlusion(self, strack_pool, occluded_val_list):
        pred_index = []
        for i in range(len(strack_pool)):
            if strack_pool[i].overlap_len == 0 and occluded_val_list[i] == True:
                if strack_pool[i].tlwh_queue.length() == self.obs_seq_len*self.frames_per_set:   
                    strack_pool[i].overlap_len += 1
                    pred_index.append(i)
                
            elif strack_pool[i].overlap_len != 0 and occluded_val_list[i] == True:
                strack_pool[i].overlap_len += 1
                
            elif strack_pool[i].overlap_len != 0 and occluded_val_list[i] == False:
                if len(strack_pool[i].pred_traj) > 0:
                    pred_traj_index = (strack_pool[i].overlap_len)//self.frames_per_set
                    
                    if pred_traj_index >= len(strack_pool[i].pred_traj):
                        pred_traj_index = len(strack_pool[i].pred_traj) - 1
                    
                    c_x, b_y = strack_pool[i].pred_traj[pred_traj_index]
                    _,_,w,h = strack_pool[i].last_tlbr

                    strack_pool[i]._tlwh = np.asarray([c_x - (w/2), b_y - h, w, h], dtype=np.float)
                    
                strack_pool[i].pred_traj = [[]]
                strack_pool[i].overlap_len = 0
                strack_pool[i].tlwh_queue.reset()
                strack_pool[i].last_tlbr = None
            
        if len(pred_index)>0:
            strack_pool = self.trajectory_prediction(strack_pool, pred_index)
        
        return strack_pool
    
    def update(self, output_results, img):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        if len(output_results):
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]  # x1y1x2y2
            # classes = output_results[:, -1]
            
            remain_inds = scores > self.args.track_thresh
            inds_low = scores > 0.1
            inds_high = scores < self.args.track_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            
            dets_second = bboxes[inds_second]
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            scores_second = scores[inds_second]
        else:
            dets_second = []
            dets = []
            scores_keep = []
            scores_second = []
        
        # check occlusion in dets
        occluded_val_new_list = check_occlusion(dets)
        

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, occluded_val=oc) for
                          (tlbr, s, oc) in zip(dets, scores_keep, occluded_val_new_list)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        
        # Predict the current location with KF
        # STrack.multi_predict(strack_pool)
        
        ################### T P ##################
        
        # check occlusion in strack_pool
        detections_tlbr = [STrack.tlwh_to_tlbr(s._tlwh) for s in strack_pool]
        occluded_val_list = check_occlusion(detections_tlbr)

        # Process occlusion
        strack_pool = self.process_occlusion(strack_pool, occluded_val_list) # process about strack_pool's _tlwh
        
        ##########################################
        
        dists = matching.iou_distance(strack_pool, detections)
        # if not self.args.mot20:
        dists = matching.fuse_score(dists, detections)
        
        matches, u_track, u_detection = matching.linear_assignment_hungarian(dists, thresh=self.args.match_thresh)
        
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment_hungarian(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        # if not self.args.mot20:
        dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment_hungarian(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
        
            track.activate(self.frame_id)
            activated_starcks.append(track)
            
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

def fft_filter(coordinates, cutoff_freq_ratio):
    # Perform FFT
    fft_coords = fft(coordinates)
    # Remove high-frequency components
    num_freqs_to_keep = int(cutoff_freq_ratio * len(fft_coords))
    fft_coords[num_freqs_to_keep:] = 0

    filtered_coords = np.real(ifft(fft_coords))

    return filtered_coords

def average_coordinates(x_coords, y_coords, cutoff_freq_ratio, frames_per_set):
    num_sets = len(x_coords) // frames_per_set
    x_averaged_coordinates = []
    y_averaged_coordinates = []
    for i in range(num_sets):
        start = i * frames_per_set
        end = (i + 1) * frames_per_set

        # Apply FFT filter
        filtered_x = fft_filter(x_coords[start:end], cutoff_freq_ratio)
        filtered_y = fft_filter(y_coords[start:end], cutoff_freq_ratio)

        x_averaged_coordinates.append(np.median(filtered_x))
        y_averaged_coordinates.append(np.median(filtered_y))
    
    return [x_averaged_coordinates, y_averaged_coordinates]

def check_occlusion(detections_tlbr):
    # tlwh to tlbr
    if len(detections_tlbr) > 0:
        ious = matching.ious(detections_tlbr, detections_tlbr)

        occluded_val_list = []
        for iou_row in ious:
            if sum(iou_row)>1.33:
                occluded_val_list.append(True)
            else:
                occluded_val_list.append(False)
        return occluded_val_list
    else:
        return []

def calculate_v_obs(obs_traj):
    num_objects = obs_traj.shape[1]
    num_dimensions = obs_traj.shape[2]

    # Calculate the difference between consecutive position vectors
    diff = obs_traj[:, :, :, 1:] - obs_traj[:, :, :, :-1]

    # Pad the first time step with zeros to match the shape of the obs_traj tensor
    padding = torch.zeros((1, num_objects, num_dimensions, 1), dtype=torch.float32)

    # Concatenate the padding and the difference tensor
    v_obs = torch.cat((padding, diff), dim=-1)
    
    temp_total = []
    for traj in v_obs[0].tolist():
        temp = []
        for i in range(len(traj[0])):
            temp.append([traj[0][i], traj[1][i]])

        temp_total.append(temp)
    
    V_obs_final = []
    for j in range(len(temp_total[0])):
        temp = []
        for i in range(len(temp_total)):
            temp.append([temp_total[i][j][0], temp_total[i][j][1]])
        V_obs_final.append(temp)
    
    return torch.tensor(V_obs_final).unsqueeze(0)

def seq_to_nodes(seq_):
    max_nodes = seq_.shape[1]  #number of pedestrians in the graph
    seq_ = seq_[0]
    seq_len = seq_.shape[2]

    V = np.zeros((seq_len, max_nodes, 2))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_[h]
    

    return V

def nodes_rel_to_nodes_abs(nodes, init_node):
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s, ped, :] = np.sum(nodes[:s + 1, ped, :],
                                       axis=0) + init_node[ped, :]

    return nodes_

