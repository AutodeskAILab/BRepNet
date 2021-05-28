import torch
from torch.utils.data import Sampler
from random import randint

class MaxNumFacesSampler(Sampler):
    def __init__(self, data_source, max_num_faces_per_batch):
        super(MaxNumFacesSampler, self).__init__(data_source)
        self.num_faces_per_brep = self.find_num_faces_per_brep(data_source)
        self.batches = self.create_batches(self.num_faces_per_brep, max_num_faces_per_batch)

    def __len__(self):
        """
        Get the number of batches
        """
        return len(self.batches)

    def __iter__(self):
        """
        Get an iterator over the batches
        """
        return iter(self.batches)


    def find_num_faces_per_brep(self, data_source):
        """
        Find the number of faces in each brep
        """
        num_faces_per_brep = []
        num_breps = len(data_source)
        for i in range(num_breps):
            data = data_source[i]
            num_faces_per_brep.append(data["face_features"].size(0))
        assert len(num_faces_per_brep) == num_breps
        return num_faces_per_brep


    def create_batches(self, num_faces_per_brep, max_num_faces_per_batch):
        """
        Create batches with a fixed size of num_faces_per_batch.
        Here we try hard to exactly replicate the original algorithm
        used to generate batches in the paper
        """
        batches = []

        num_faces_in_current_batch = 0
        current_batch = []

        index_list = []
        for index, num_faces in enumerate(num_faces_per_brep):
            index_list.append(
                {
                    "index": index,
                    "num_faces": num_faces
                }
            )

        while len(index_list) > 0:
            # Pick an item from index list at random
            random_index = randint(0, len(index_list)-1)
            random_element = index_list[random_index]

            # Shuffle the last element to the free slot.
            # It's a C++ trick and not good python practice...
            last_element = index_list.pop()
            if random_index < len(index_list):
                index_list[random_index] = last_element

            num_faces = random_element["num_faces"]
            random_element_index = random_element["index"]

            # Can we add this solid to the batch
            if num_faces_in_current_batch + num_faces > max_num_faces_per_batch:
                if len(current_batch)>0:
                    batches.append(current_batch)
                    current_batch = [ random_element_index ]
                    num_faces_in_current_batch = num_faces
                else:
                    assert num_faces_in_current_batch == 0
                    # Here we have the case where one single B-Rep has
                    # more than the maximum number of faces
                    print(f"Warning!!! - B-Rep {random_element_index} has {num_faces} and will not be included")
            else:
                current_batch.append(random_element_index)
                num_faces_in_current_batch += num_faces

        if len(current_batch)>0:
            batches.append(current_batch)

        print(f"Mean num breps per batch {len(num_faces_per_brep)/len(batches)}")

        return batches
