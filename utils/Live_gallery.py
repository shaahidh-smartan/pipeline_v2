import torch
import threading

class LiveGallery:
    def __init__(self, device):
        self.device = device
        self.lock = threading.Lock()

        # person id -> {"embedding" : Tensor, "visibility": Tensor}
        self.pid_to_data = {}

        # cache the flattened embeds
        self.emb = None
        self.vis = None
        self.pids = None
        self.dirty = True

    def add_person(self, pid, embeddings, visibility):
        embeddings = embeddings.to(self.device)
        visibility = visibility.to(self.device)

        with self.lock:
            self.pid_to_data[pid] = {
                "embeddings": embeddings,
                "visibility": visibility
            }
            self.dirty = True

    def remove_person(self, pid):
        with self.lock:
            if pid in self.pid_to_data:
                del self.pid_to_data[pid]
                self._dirty = True

    def get_flat_gallery(self):
        with self.lock:
            if not self.pid_to_data:
                return None, None, None

            if self._dirty:
                emb_list, vis_list, pid_list = [], [], []

                for pid, data in self.pid_to_data.items():
                    n = data["embeddings"].shape[0]
                    emb_list.append(data["embeddings"])
                    vis_list.append(data["visibility"])
                    pid_list.append(
                        torch.full(
                            (n,),
                            pid,
                            dtype=torch.long,
                            device=self.device
                        )
                    )

                self.emb = torch.cat(emb_list, dim=0)
                self.vis = torch.cat(vis_list, dim=0)
                self.pids = torch.cat(pid_list, dim=0)
                self.dirty = False

            return self.emb, self.vis, self.pids
    