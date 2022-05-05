import sys
import cv2
import numpy as np
import json
import time
import datetime
import matplotlib.pyplot as plt


height, width = 768, 768 
font = cv2.FONT_HERSHEY_SIMPLEX

red = np.ones((height, width, 3)).astype(np.uint8)
green = np.ones((height, width, 3)).astype(np.uint8)
yellow = np.ones((height, width, 3)).astype(np.uint8)
gray = np.ones((height, width, 3)).astype(np.uint8)

red[:, :, 2] *= 255
green[:, :, 1] *= 255
yellow[:, :, 1:3] *= 255
gray[:, :, :] *= 128


# GO = gray.copy()
# GO = cv2.putText(GO, "V", (384, 384), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
# STOP = gray.copy()
# STOP = cv2.putText(STOP, "V", (384, 384), font, 1.5, (147, 0, 214), 2, cv2.LINE_AA)
# DECIDE = gray.copy()
# DECIDE = cv2.putText(DECIDE, "V", (384, 384), font, 1.5, (255, 102, 102), 2, cv2.LINE_AA)

# larger V, same as abstract.py
GO = gray.copy()
GO = cv2.putText(GO, "V", (340, 380), font, 5, (255, 255, 255), 5, cv2.LINE_AA)
STOP = gray.copy()
STOP = cv2.putText(STOP, "V", (340, 380), font, 5, (147, 0, 214), 5, cv2.LINE_AA)
DECIDE = gray.copy()
DECIDE = cv2.putText(DECIDE, "V", (340, 380), font, 5, (255, 102, 102), 5, cv2.LINE_AA)

class Experiment:
    experiment_id = ""
    title = ""
    filename = ""

    # (go, stop, decide) must sum to 1.0
    proportions = (0.50, 0.25, 0.25)  # for experiment
    # proportions = (0.33, 0.33, 0.34)  # for presentation purpose

    records = []
    count = 0
    delay_stop = 300
    delay_decide = 300

    font = cv2.FONT_HERSHEY_SIMPLEX
   
    def __init__(self, experiment_id):
        self.experiment_id = experiment_id + "-" + datetime.datetime.now().strftime("%c")
        self.experiment_id = self.experiment_id.replace(" ", "-")
        self.experiment_id = self.experiment_id.replace(":", "-")
        self.title = f"Experiment: {self.experiment_id}"
        self.filename = f"./data/{self.experiment_id}.json"
    
    def run(self):
        while True:
            img = gray.copy()
            img = cv2.putText(img, "Press C => Start a round of experiments", (40, 364), self.font, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
            img = cv2.putText(img, "Press Q => Quit and store the experiment records", (40, 394), self.font, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
            img = cv2.putText(img, "Completed rounds: " + str(self.count), (40, 424), self.font, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow(self.title, img)

            k = cv2.waitKey(0)
            while k != ord('q') and k != ord('c'):
                k = cv2.waitKey(0)
            
            if (k == ord('c')):
                self.one_experiment()
                continue

            if (k == ord('q')):
                self.save()
                self.plot([self.filename])
                break


    def one_experiment(self):
        self.count += 1
        rec = None
        r = np.random.rand()
        experiment_type = "go" if r <= self.proportions[0] else ("stop" if (r - self.proportions[0]) <= self.proportions[1] else "decide")
        img = gray.copy()
        img = cv2.putText(img, "x", (384, 384), self.font, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow(self.title, img)
        cv2.waitKey(500)
        cv2.imshow(self.title, gray)
        cv2.waitKey(500)

        if experiment_type == 'go':
            cv2.imshow(self.title, GO)
            start_time = time.time()
            k = cv2.waitKey(2000)
            end_time = time.time()
            img = gray.copy()
            if k == -1:  # no response
                self.count -= 1
                img = cv2.putText(img, "Error. Please response ASAP!", (40, 354), self.font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                img = cv2.putText(img, "This round is discarded.", (40, 384), self.font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                img = cv2.putText(img, "Press C to continue.", (40, 414), self.font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow(self.title, img)
                while cv2.waitKey(0) != ord('c'):
                    pass
            else:   # receive response within 2 sec
                response_time = (end_time - start_time) * 1000
                rec = self.record(experiment_type, response_time)
                img = cv2.putText(img, "Succeed. The response time is recorded", (20, 384), self.font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                img = cv2.putText(img, "Press C to continue.", (20, 414), self.font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow(self.title, img)
                while cv2.waitKey(0) != ord('c'):
                    pass
        elif experiment_type == 'stop':
            cv2.imshow(self.title, GO)
            start_time = time.time()
            k = cv2.waitKey(self.delay_stop)
            end_time = time.time()
            if k != -1:     # respond before stop sign
                response_time = (end_time - start_time) * 1000
                rec = self.record(experiment_type, response_time)
                self.delay_stop -= 20
                img = gray.copy()
                img = cv2.putText(img, "Succeed. The response time is recorded", (20, 384), self.font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                img = cv2.putText(img, "Press C to continue.", (20, 414), self.font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow(self.title, img)
                while cv2.waitKey(0) != ord('c'):
                    pass
            else:   # show stop sign
                cv2.imshow(self.title, STOP)
                k = cv2.waitKey(2000 - self.delay_stop)
                end_time = time.time()
                if k != -1:     # stop failed
                    response_time = (end_time - start_time) * 1000
                    rec = self.record(experiment_type, response_time)
                    self.delay_stop -= 20
                    img = gray.copy()
                    img = cv2.putText(img, "Stop failed. The response time is recorded", (20, 384), self.font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    img = cv2.putText(img, "Press C to continue.", (20, 414), self.font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.imshow(self.title, img)
                    while cv2.waitKey(0) != ord('c'):
                        pass
                else:   # stop Succeed
                    self.delay_stop += 20
                    img = gray.copy()
                    img = cv2.putText(img, "Stop succeed.", (20, 384), self.font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    img = cv2.putText(img, "Press C to continue.", (20, 414), self.font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.imshow(self.title, img)
                    while cv2.waitKey(0) != ord('c'):
                        pass
        else:
            cv2.imshow(self.title, GO)
            start_time = time.time()
            k = cv2.waitKey(self.delay_decide)
            end_time = time.time()
            if k != -1:     # respond before decide sign
                response_time = (end_time - start_time) * 1000
                rec = self.record(experiment_type + "-fail", response_time)
                self.delay_decide -= 20
                img = gray.copy()
                img = cv2.putText(img, "Succeed. The response time is recorded", (20, 384), self.font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                img = cv2.putText(img, "Press C to continue.", (20, 414), self.font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow(self.title, img)
                while cv2.waitKey(0) != ord('c'):
                    pass
            else:   # show decide sign
                cv2.imshow(self.title, DECIDE)
                k = cv2.waitKey(2000 - self.delay_decide)
                end_time = time.time()
                if k != -1:     # button pressed in a decide round
                    response_time = (end_time - start_time) * 1000
                    img = gray.copy()
                    img = cv2.putText(img, "Button pressed in a DECIDE round.", (20, 354), self.font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    img = cv2.putText(img, "Press Y if it was your own decision", (20, 384), self.font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    img = cv2.putText(img, "Press N if you were just acting on impulse", (20, 414), self.font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.imshow(self.title, img)

                    k = cv2.waitKey(0)
                    while k != ord('y') and k != ord('n'):
                        k = cv2.waitKey(0)

                    if k == ord('y'):
                        rec = self.record(experiment_type + "-succeed", response_time)
                        self.delay_decide += 20
                    else:
                        rec = self.record(experiment_type + "-fail", response_time)
                        self.delay_decide -= 20

                    img = gray.copy()
                    img = cv2.putText(img, "The response time is recorded", (20, 384), self.font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    img = cv2.putText(img, "Press C to continue.", (20, 414), self.font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.imshow(self.title, img)
                    while cv2.waitKey(0) != ord('c'):
                        pass
                else:   # no button pressed in a decide round
                    self.delay_decide += 20
                    img = gray.copy()
                    img = cv2.putText(img, "You decide not to press the button.", (20, 384), self.font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    img = cv2.putText(img, "Press C to continue.", (20, 414), self.font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.imshow(self.title, img)
                    while cv2.waitKey(0) != ord('c'):
                        pass
        
        img = gray.copy()
        img = cv2.putText(img, "Press S to save this record", (40, 374), self.font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        img = cv2.putText(img, "Press N to discard this record", (40, 404), self.font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow(self.title, img)
        k = cv2.waitKey(0)
        while k != ord('s') and k != ord('n'):
            k = cv2.waitKey(0)

        if k == ord('s') and rec is not None:
            self.records.append(rec)



    
    def record(self, expm_type, response_time):
        assert(expm_type in ['go', 'stop', 'decide-succeed', 'decide-fail'])
        return {
            "id": self.experiment_id,
            "type": expm_type,
            "time": response_time
        }


    def save(self):
        with open(self.filename, 'w') as f:
            json.dump(self.records, f)
    
    @staticmethod
    def plot(fn_list):
        records = []
        for fn in fn_list:
            with open(fn) as f:
                data = json.load(f)
                for d in data:
                    assert(d["type"] in ['go', 'stop', 'decide-succeed', 'decide-fail'])
                    records.append(d)
        
        go_t = [d["time"] for d in records if d["type"] == 'go']
        stop_f_t = [d["time"] for d in records if d["type"] == 'stop']
        decide_s_t = [d["time"] for d in records if d["type"] == 'decide-succeed']
        decide_f_t = [d["time"] for d in records if d["type"] == 'decide-fail']

        t = [go_t, stop_f_t, decide_s_t, decide_f_t]
        labels = ["Go", "Stop Failed", "Decide Succeeded", "Decide Failed"]

        plt.figure()

        plt.hist(t, bins=20, range=(200, 1500), label=labels)
        plt.legend(prop ={'size': 10})
        plt.ylabel("Counts")
        plt.xlabel("Reaction Time (ms)")
        plt.show()
        

        

if __name__ == '__main__':
    argc = len(sys.argv)
    if argc == 1:
        experiment_id = input("Please enter an identifier for this experiment: \n")
        ex = Experiment(experiment_id)
        ex.run()
    else:
        fns = sys.argv[1:]
        Experiment.plot(fns)
    