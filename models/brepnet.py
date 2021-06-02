from collections import OrderedDict
import numpy as np
from pathlib import Path
from pytorch_lightning.core.lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.data_utils as data_utils
from dataloaders.brepnet_dataset import BRepNetDataset, brepnet_collate_fn
from dataloaders.brepnet_dataset_old import BRepNetDatasetOld
from dataloaders.max_num_faces_sampler import MaxNumFacesSampler


def build_matrix_Psi(Xf, Xe, Xc, Kf, Ke, Kc):
    """
    Build the matrix Psi.

    Please see equation (2) in the BRepNet paper.

    For each coedge in the model we know the indices of some 
    neighboring faces, edges and coedges.   These indices
    are in the rows of the integer tensors Kf, Ke and Kc.

    We use these indices to extract the appropriate rows of
    the feature matrices Xf, Xe and Xc.  These are then 
    concatenated to build the matrix Psi.

    The input consists of the feature tensors

    Xf.size() = [ num_faces x num_face_features ]
    Xe.size() = [ num_edges x num_edge_features ]
    Xc.size() = [ num_coedges x num_coedge_features ]

    and the index tensors

    Kf.size() = [ num_coedges x num_faces_in_kernel ]
    Ke.size() = [ num_coedges x num_edges_in_kernel ]
    Kc.size() = [ num_coedges x num_coedges_in_kernel ]
    """
    # First we use the kernel index matrices to pull the 
    # appropriate rows of Xf, Xe and Xc into 3 tensors
    
    # Pft.size() = [ num_coedges x num_faces_in_kernel x num_face_features ]
    Pft = Xf[Kf]

    # Pet.size() = [ num_coedges x num_edges_in_kernel x num_edge_features ] 
    Pet = Xe[Ke]

    # Pct.size() = [ num_coedges x num_coedges_in_kernel x num_coedge_features ] 
    Pct = Xc[Kc]

    # Next we need to flatten these tensors to give tensors of size
    # [ num_coedges x (num_ents*num_ent_features) ]
    Pt = torch.flatten(Pft, 1)
    Pe = torch.flatten(Pet, 1)
    Pc = torch.flatten(Pct, 1)

    # Now we can concatentate these tensors to form Psi 
    Psi = torch.cat([Pt, Pe, Pc], dim=1)

    return Psi


def find_max_feature_vectors_for_each_edge(Ze, Ce):
    """
    Each edge in the B-Rep has two coedges.  In this function
    we perform an element-wise max pooling of the feature vectors
    for the two coedges to produce a feature vector for the edges.

    Please see equation (4) in the BRepNet paper.

    The input consists of 

    Ze.size() = [ num_coedges x num_filters ]
    Ce.size() = [ num_edges x 2]
    """
    # For the tensor Ze we need to take the max feature vector
    # values for the two coedges.   First we build
    # zet.size() =  [ num_edges x 2 x num_filters ]
    Zet = Ze[Ce]

    # Now we can take the max along dim 1
    (He, Heargmax) = torch.max(Zet, dim=1)

    return He

def find_max_feature_vectors_for_each_face(Zf, Cf, Csf, device):
    """
    Each face in the B-Rep will have many coedges. In this function
    we perform an element-wise max pooling of the coedge features in
    matrix Zf, to create a matrix of face feature vectors Hf.

    Please see equation (4) in the BRepNet paper.

    We have a matrix Zf with size [ num_coedges x num_filters ].

    For faces with less than or equal to max_coedges coedges 
    in the loops the indices of the coedges are in 

    Cf.size() = [ num_small_faces x max_coedges ]
    
    The tensor is padded with the index num_coedges.  A row of
    zeros will be concatenated to the mlp output tensor to allow 
    this padding to work.

    For faces with more than max_coedges coedges we have
    an array of tensors of different sizes
    
    Csf = [
        Csf.size() = [ num_coedges_in_face_1 ],
        ...
    ]
    """
    # There can be many coedges around each face, so rather than having one 
    # index tensor we have things split into two bits.

    # We add a row of padding to Zf.  The tensor Cf will index into 
    # this padding when a face has less than max_coedges coedges
    num_filters = Zf.size()[1]
    Zfpad = torch.cat([Zf, torch.zeros(1, num_filters, device=device)],  dim=0)

    # We can now build a tensor Zft.size() = [ num_small_faces x max_coedges x num_filters ]
    Zft = Zfpad[Cf]

    # And then we take the max along dim 1 as for the edge case.  The resulting tensor
    # has size [ num_small_faces x num_filters ]
    (Hfsmall_faces, Hfsmall_faces_argmaxs)  = torch.max(Zft, dim=1)

    # Now for the faces with many halfedges we have to do this face by face
    Hf_array = [ Hfsmall_faces ]
    for Csingle_face in Csf:
        # Create tensor for a single face
        # Zsingle_face.size() = [ num_coedges_on_ith_face x num_filters ]
        Zsingle_face = Zf[Csingle_face]

        # Now we take the max over the coedges in dim 0 to gave us
        # Hbig_face.size() = [  num_filters ]
        # We need to reshape it to  [ 1 x num_filters ]
        (Hbig_face, Hbig_face_argmaxs)  = torch.max(Zsingle_face, dim=0)
        Hf_array.append(torch.reshape(Hbig_face, (1, num_filters)))

    # Now we can create the final Hf by concatenating all the Hf tensors
    # The tensor Hf will now have size [ num_faces x num_filters ]
    Hf = torch.cat(Hf_array, dim=0)

    return Hf



class BRepNetMLP(LightningModule):
    """
    The MLP at the heart of BRepNet.

    This is where all the learnable parameters live.  
    
    The first layer of the MLP always needs to consume the
    rows of Psi.  This is the MLP input size.

    The intermediate layers of the MLP have hidden_size.

    The output layer of the MLP has output_size.

    If this is the final layer in the entire network then
    we do not add a ReLU or bias vector.  Adding a final
    ReLU has the effect of preventing the logits from having
    negative values, which greatly reduces performance. 
    """

    def __init__(self, num_layers, input_size, hidden_size, output_size, final_layer, dropout=None):
        """Initialize the layer"""
        super(BRepNetMLP, self).__init__()
        assert num_layers > 0, "Must have at least 1 layer"

        mlp_layers = OrderedDict()
        for i in range(0,num_layers):
            is_first_mpl_layer = (i==0)
            is_last_mlp_layer = (i==num_layers-1)

            # General case
            use_bias = True
            use_relu = True
            linear_input_size = hidden_size
            linear_output_size = hidden_size

            # First layer
            if is_first_mpl_layer:
                linear_input_size = input_size

            # Last layer
            if is_last_mlp_layer:
                linear_output_size = output_size

                if final_layer:
                    # For the very last layer in the network we don't want
                    # to use a bias or ReLU
                    use_bias = False
                    use_relu = False

            mlp_layers[f"linear_{i}"] = nn.Linear(linear_input_size, linear_output_size, bias=use_bias)

            if dropout is not None:
                mlp_layers[f"dropout_{i}"] = nn.Dropout(p=dropout)

            if use_relu:
                mlp_layers[f"relu_{i}"] = nn.ReLU()
        
        self.mlp = nn.Sequential(mlp_layers)
            
    def forward(self, Psi):
        """Forward pass through the MLP"""
        return self.mlp(Psi)


class BRepNetLayer(LightningModule):
    """
    A general layer in BRepNet.  
    This can be either the input layer or one of the hidden layers.
    """

    def __init__(self, num_mlp_layers, input_size, output_size, dropout=None):
        """
        Initialization of a general BRepNet layer.

        num_mlp_layers - Number of layers we want in the MLP

        input_size     - This needs to be set to the total length of all the feature vectors
                         which will take part in the convolution

        output_size    - This needs to be set to the length of the output feature vectors
                         for the faces, edges and coedges

        dropout        - To use dropout, set this to the dropout probablity
                         No dropout is used if this is set to None
        """ 
        super(BRepNetLayer, self).__init__()
        self.output_size = output_size

        # This is not the final layer
        final_layer = False

        # The matrix will get split into 3 components, one for faces, one for edges and 
        # one for coedges.  Hence the output of the MLP should always be 3 times
        # the final output size.
        self.mlp = BRepNetMLP(num_mlp_layers, input_size, 3*output_size, 3*output_size, final_layer, dropout)
        

    def forward(self, Xf, Xe, Xc, Kf, Ke, Kc, Ce, Cf, Csf):
        """
        This layer performs the following steps

            1) The matrix Psi is by permuting and concatenating the
               feature vectors in Xf, Xe and Xc

            2) Psi is fed through an MLP.  This is where all the
               learnable parameters in the network live.  This is
               convolution implemented with general matrix 
               multiplication (GEMM).  See Hanocka et al.

            3) The output from the MLP, Z is split into
               Zf, Ze, Zc

            4) The coedges features in Zf and Ze are max pooled
               onto the edges and faces
        """

        # We use the kernel index matrices to construct a matrix Psi with
        # size [ num_coedges x mlp_input_size]
        Psi = build_matrix_Psi(Xf, Xe, Xc, Kf, Ke, Kc)

        # Next the mlp is applied to Psi
        Z = self.mlp(Psi)

        # Now we need to split Z into 3 parts
        Zc = Z[:, : self.output_size]
        Ze = Z[:, self.output_size : 2*self.output_size]
        Zf = Z[:, 2*self.output_size : ]

        # The tensor Zc is now the output Hc

        # Each edge has two coedges.  We need to find the 
        # maximum of the two feature vectors for each edge
        He = find_max_feature_vectors_for_each_edge(Ze, Ce)

        # Finally we need to do the same thing for faces
        Hf = find_max_feature_vectors_for_each_face(Zf, Cf, Csf, self.device)

        return (Hf, He, Zc, Kf, Ke, Kc, Ce, Cf, Csf)


class BRepNetFaceOutputLayer(LightningModule):
    """
    The output layer in the network for face classification.

    This layer is very similar to the general layer.  The key difference
    is that it will generate only the logits for the face classifications.
    The hidden state for edges and coedges will not be created.
    """
        
    def __init__(self, num_mlp_layers, input_size, hidden_size, output_size, dropout=None):
        """
        Initialization of the BRepNet output layer.

        num_mlp_layers - Number of layers we want in the MLP

        input_size     - This needs to be set to the total length of all the feature vectors
                         which will take part in the convolution

        hidden_size    - This should generally be set to the MLP hidden size used in 
                         the rest of the network.

        output_size    - This needs to be the number of classes which faces 
                         can be categorized into.

        dropout        - To use dropout, set this to the dropout probablity
                         No dropout is used if this is set to None
        """ 
        super(BRepNetFaceOutputLayer, self).__init__()

        # This is the final layer of the network.  We need to pass this
        # flag to the MLP so it knows that the final ReLU and bias are 
        # not required
        final_layer = True

        # The output layer has the same hidden size as all the other layers
        # but the output size is just the number of classes
        self.mlp = BRepNetMLP(num_mlp_layers, input_size, 3*hidden_size, output_size, final_layer, dropout)


    def forward(self, Xf, Xe, Xc, Kf, Ke, Kc, Ce, Cf, Csf):
        """
        This layer performs the following steps

            1) The matrix Psi is by permuting and concatenating the
               feature vectors in Xf, Xe and Xc

            2) Psi is fed through an MLP.  This is where all the
               learnable parameters in the network live.  This is
               convolution implemented with general matrix 
               multiplication (GEMM).  See Hanocka et al.

            3) The coedges features in Z are max pooled
               to provide the logits for the faces
        """
        
        # We use the kernel index matrices to construct a matrix Psi with
        # size [ num_coedges x mlp_input_size]
        Psi = build_matrix_Psi(Xf, Xe, Xc, Kf, Ke, Kc)

        # Next the mlp is applied to Psi
        Z = self.mlp(Psi)

        # Finally use max pooling to combine the coedge
        # activations in Z to build the logits for faces
        Hf = find_max_feature_vectors_for_each_face(Z, Cf, Csf, self.device)
        return Hf



class BRepNet(LightningModule):
    """The main BRepNet network"""

    def __init__(self, opts):
        """
        Initialization of the main BRepNet network.

        """
        super(BRepNet, self).__init__()
        self.opts = opts
        kernel = data_utils.load_json_data(opts.kernel)
        input_feature_metadata = data_utils.load_json_data(opts.input_features)
        num_classes = opts.num_classes

        # Set up the names of the segments for clearer 
        # output statistics
        segment_names_file = self.find_segment_names_file(opts)
        if segment_names_file is not None:
            self.segment_names = data_utils.load_json_data(segment_names_file)

        # We always have one special input and special output layer
        assert opts.num_layers >= 2

        num_mlp_layers = opts.num_mlp_layers
        num_filters = opts.num_filters

        # The size of the matrix Psi depends on the number of entities 
        # in the kernel.  For the very first layer it also depends on
        # the length of the input feature vectors for each entity.
        num_faces_per_kernel = len(kernel["faces"])
        num_edges_per_kernel = len(kernel["edges"])
        num_coedges_per_kernel = len(kernel["coedges"])
        num_face_features = len(input_feature_metadata["face_features"])
        num_edge_features = len(input_feature_metadata["edge_features"])
        num_coedge_features = len(input_feature_metadata["coedge_features"])
        
        # The very first layer of the network needs to ingest a concatenated
        # set of feature vectors based on the number of features extracted from the
        # geometry. 
        mlp_input_size = (num_faces_per_kernel*num_face_features) + \
                         (num_edges_per_kernel*num_edge_features) + \
                         (num_coedges_per_kernel*num_coedge_features)

        # Subsequently the size depends only on the number of filters and
        # number of faces, edges and coedges in the kernel
        mlp_hidden_size = num_filters*(num_faces_per_kernel + num_edges_per_kernel + num_coedges_per_kernel)

        # Create the layers of the network
        self.layers = nn.ModuleList()

        dropout = opts.dropout

        # The first layer has a size based on in the number of input features
        self.layers.append(BRepNetLayer(num_mlp_layers, mlp_input_size, num_filters, dropout))

        # The hidden layers has a size based on in the number of filters
        for l in range(2, opts.num_layers):
            self.layers.append(BRepNetLayer(num_mlp_layers, mlp_hidden_size, num_filters, dropout))

        # The output layer is similar, but it generates only
        # the logits for the faces.
        self.output_layer = BRepNetFaceOutputLayer(num_mlp_layers, mlp_hidden_size, num_filters, num_classes, dropout) 

        # Save the hyper-parameters
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--dataset_file", type=str, required=True, help="Path to the dataset file containing the train/val/test split")
        parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory the files generated by pextract_brepnet_data_from_step.py")
        parser.add_argument("--label_dir",  type=str, help="Path to the directory containing the segmentation labels.  This will typically be the step dir in the dataset")
        parser.add_argument("--log_dir",  type=str, default="./logs", help="Path to the directory where you want to write logs")
        parser.add_argument("--input_features", type=str, default="feature_lists/all.json", help="List of features to read")
        parser.add_argument("--kernel", type=str, default="kernels/winged_edge.json", help="Which kernel to use")
        parser.add_argument("--dropout", type=float, help="If using dropout then this is the dropout probability")
        parser.add_argument("--segment_names", type=str, help="The segment names file from the dataset")
        parser.add_argument("--num_layers", type=int, default=2, help="2 gives just the input and output layers")
        parser.add_argument("--num_mlp_layers", type=int, default=2, help="Number of layers in the mlp.  Value > 0")
        parser.add_argument("--num_filters", type=int, default=84, help="Number of filters.  Hyper-parameter s in the paper.  Value > 0")
        parser.add_argument("--num_classes", type=int, default=8, help="Number of classes used in the dataset")
        parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
        parser.add_argument('--batch_size', type=int, default=200, help="Number of breps in one batch")
        parser.add_argument('--max_num_faces_per_batch', type=int, help="If defined this sets a limit on the number of faces per batch")
        parser.add_argument('--num_workers', type=int, default=0, help="Number of worker threads")
        parser.add_argument('--use_old_dataloader', action="store_true", help="Use the old dataloader")
        parser.add_argument('--shuffle_train_set', type=bool, default=True, help="Use shuffling on the training set")
        parser.add_argument("--test_with_validation_set", action="store_true", help="Model to use for testing")
        parser.add_argument("--logit_dir", type=str, help="Save logits to this directory")
        return parser


    def find_segment_names_file(self, opts):
        """
        Try to find the segment names file in some usual places
        """
        if opts.segment_names is not None:
            segment_names_file = Path(opts.segment_names)
            if segment_names_file.exists():
                return segment_names_file
            else:
                print(f"Warning! {segment_names_file}  not found")
        
        # Try looking for the segment names file above the dataset_dir
        segment_names_file = Path(opts.dataset_dir).parent / "segment_names.json"
        if segment_names_file.exists():
                return segment_names_file

        print("Warning! segment names not found.")
        print("Use the option --segment_names path/to/segment_names.json")
        return None
            

    def forward(self, Xf, Xe, Xc, Kf, Ke, Kc, Ce, Cf, Csf):
        """
        A forward pass through the network.
        """
        for i, layer in enumerate(self.layers):
            Xf, Xe, Xc, Kf, Ke, Kc, Ce, Cf, Csf = layer(Xf, Xe, Xc, Kf, Ke, Kc, Ce, Cf, Csf)

        return self.output_layer(Xf, Xe, Xc, Kf, Ke, Kc, Ce, Cf, Csf)


    def total_num_parameters(self):
        """
        Find the total number of parameters in the network.
        """
        num_params = 0
        for p in self.parameters():
            nn = 1
            for s in list(p.size()):
                nn = nn*s
            num_params += nn
        return num_params


    def print_parameter_info(self):
        """
        Print more detailed info about the network parameters.
        """
        for name, p in self.named_parameters():
            print(f"{name}:  {p.size()}")


    def find_loss(self, logits, all_batch_labels):
        """Find the loss given the logits and labels"""
        return F.cross_entropy(logits, all_batch_labels, reduction='mean')

    
    def find_predicted_classes(self, t):
        """
        Find the predicted classes from the un-normalized segmentation scores
        """
        norm_seg_scores = F.softmax(t.detach(), dim=1)
        return torch.argmax(norm_seg_scores, dim=1)


    def brepnet_step(self, batch, batch_idx, save_logits):
        """
        A train or validation step for the BRepNet network on one batch
        """
        # Unpack the tensor data
        Xf = batch["face_features"]
        Xe = batch["edge_features"]
        Xc = batch["coedge_features"]
        Kf = batch["face_kernel_tensor"]
        Ke = batch["edge_kernel_tensor"]
        Kc = batch["coedge_kernel_tensor"]
        Ce = batch["coedges_of_edges"]
        Cf = batch["coedges_of_small_faces"]
        Csf = batch["coedges_of_big_faces"]

        # Make the forward pass through the network
        # The tensor logits is now size [ num_faces_in_batch x num_classes ]
        segmentation_scores = self(Xf, Xe, Xc, Kf, Ke, Kc, Ce, Cf, Csf)

        # We may want to save the logits for use in downstream procedures
        # like visualization or CAD automation.  We save them here is requested
        if save_logits:
            self.save_logits(batch, segmentation_scores.detach())

        # Now find the loss
        labels = batch["labels"]
        loss = self.find_loss(segmentation_scores, labels)

        # Find the network predictions
        predicted_classes = self.find_predicted_classes(segmentation_scores)

        # Compute the accuracy for the logs
        num_faces = labels.size(0)
        num_labels_per_face = segmentation_scores.size(1)
        assert segmentation_scores.size(0) == num_faces, "Must have same number of faces"
        correct = (labels==predicted_classes)
        num_faces_correct = torch.sum(correct).item()
        accuracy = num_faces_correct/num_faces

        # Compute the per-class IoU
        per_class_intersections = [0.0] * self.opts.num_classes
        per_class_unions = [0.0] * self.opts.num_classes
        for i in range(num_labels_per_face):
            selected = (predicted_classes == i)
            selected_correct = (selected & correct)
            labelled = (labels == i)
            union = selected | labelled
            per_class_intersections[i] += selected_correct.sum().item()
            per_class_unions[i] += union.sum().item()

        iou_data = {
            "num_faces": num_faces,
            "num_faces_correct": num_faces_correct,
            "per_class_intersections": per_class_intersections,
            "per_class_unions": per_class_unions
        }

        return {
            "loss": loss,
            "accuracy": accuracy,
            "iou_data": iou_data
        }
        

    def training_step(self, batch, batch_idx):
        output = self.brepnet_step(batch, batch_idx, False)
                
        # Log some data to tensorboard
        self.log("loss", output["loss"].item(), on_step=True, on_epoch=False)
        self.log("train/loss", output["loss"].item(), on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
        self.log("train/accuracy", output["accuracy"], on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
        return output["loss"]


    def validation_step(self, batch, batch_idx):
        """
        Validate one batch
        Here we call the training step and then rename the 
        keys so the logs are correct
        """
        save_logits = self.opts.logit_dir is not None
        output = self.brepnet_step(batch, batch_idx, save_logits)
        self.log("validation/loss", output["loss"].item(), on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
        return output


    def collate_epoch_outputs(self, outputs):
        """
        Collate information from all batches at the end of an epoch
        """
        num_faces_correct = 0
        total_num_faces = 0
        per_class_intersections = [0.0] * self.opts.num_classes
        per_class_unions = [0.0] * self.opts.num_classes
        for output in outputs:
            total_num_faces += output["iou_data"]["num_faces"]
            num_faces_correct += output["iou_data"]["num_faces_correct"]
            for i in range(self.opts.num_classes):
                per_class_intersections[i] += output["iou_data"]["per_class_intersections"][i]
                per_class_unions[i] += output["iou_data"]["per_class_unions"][i]

        per_class_iou = []
        mean_iou = 0.0
        for i in range(self.opts.num_classes):
            if per_class_unions[i] > 0.0:
                iou = per_class_intersections[i]/per_class_unions[i]
            else:
                # Should never come here with the full dataset
                iou = 1.0
            per_class_iou.append(iou)
            mean_iou += iou

        accuracy = num_faces_correct / total_num_faces
        mean_iou /= self.opts.num_classes
        return {
            "accuracy": accuracy,
            "mean_iou": mean_iou,
            "per_class_iou": per_class_iou
        }


    def validation_epoch_end(self, outputs):
        """
        Collate information from all validation batches
        """
        output = self.collate_epoch_outputs(outputs)
        self.log("validation/accuracy", output["accuracy"], on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
        self.log("validation/mean_iou", output["mean_iou"], on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)

        # If the segment names information is provided then log the 
        # per-class IoU
        if self.segment_names is not None:
            assert len(self.segment_names) == len(output["per_class_iou"])
            for name, iou in zip(self.segment_names, output["per_class_iou"]):
                log_name = f"validation/{name}_iou"
                self.log(log_name, iou, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)

    def test_step(self, batch, batch_idx):
        """
        Test on one batch
        """
        save_logits = self.opts.logit_dir is not None
        return self.brepnet_step(batch, batch_idx, save_logits)
                

    def test_epoch_end(self, outputs):
        """
        Collate the results from all test batches
        """
        output = self.collate_epoch_outputs(outputs)

        self.log("test/accuracy", output["accuracy"], on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
        self.log("test/mean_iou", output["mean_iou"], on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)

        # If the segment names information is provided then log the 
        # per-class IoU
        per_class_iou = {}
        if hasattr(self, "segment_names"):
            assert len(self.segment_names) == len(output["per_class_iou"])
            for name, iou in zip(self.segment_names, output["per_class_iou"]):
                log_name = f"test/{name}_iou"
                self.log(log_name, iou, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
                per_class_iou[name] = iou
            output["per_class_iou"] = per_class_iou
        return output


    def save_logits(self, batch, batch_face_seg_scores):
        """
        Save logits for this batch
        """
        output_folder = Path(self.opts.logit_dir)
        if not output_folder.exists():
            output_folder.mkdir()

        # We need to split the logits based on the 
        # split_batch info.  This splits up the logits 
        # into tensors for each solid
        for split_solid, file_stem in zip(batch["split_batch"], batch["file_stems"]):
            face_seg_scores_for_solid = batch_face_seg_scores[split_solid["face_indices"]]

            # The segmentation scores are not normalized.  We want to convert these
            # to logits (probabilities that a face is of each class)
            face_logits_for_solid = F.softmax(face_seg_scores_for_solid.detach(), dim=1)

            # Now find the pathname to save the logits file
            output_pathname = output_folder / (file_stem + ".logits")

            # Finally use numpy to save the logits information in text format
            np.savetxt(output_pathname, face_logits_for_solid.numpy())
            



    def train_dataloader(self):
        if self.opts.use_old_dataloader:
            # Legacy dataloader for json data extracted with 
            # proprietary code  
            dataset = BRepNetDatasetOld(self.opts, "training_set")
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=None  # Solids are organized into fixed batches
            )

        # Dataloader to read from open source based pipeline
        dataset = BRepNetDataset(self.opts, "training_set")

        batch_sampler = None
        shuffle = self.opts.shuffle_train_set
        batch_size = self.opts.batch_size
        if self.opts.max_num_faces_per_batch is not None:
            print("Warning! - max_num_faces_per_batch option may not work with multi-gpu or multi-node training")
            batch_sampler = MaxNumFacesSampler(dataset, self.opts.max_num_faces_per_batch)

            if shuffle:
                print("Warning! - Overriding shuffle option")
            shuffle = False

            if batch_size != 1:
                print("Warning! - Overriding batch_size option")
            batch_size = 1

        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=brepnet_collate_fn,
            batch_sampler=batch_sampler,
            batch_size=batch_size,
            num_workers=self.opts.num_workers,
            shuffle=shuffle
        )


    def val_dataloader(self):
        if self.opts.use_old_dataloader:
            # Legacy dataloader for json data extracted with 
            # proprietary code  
            dataset = BRepNetDatasetOld(self.opts, "validation_set")
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=None  # Solids are organized into fixed batches
            )

        # Dataloader to read from open source based pipeline
        dataset = BRepNetDataset(self.opts, "validation_set")
        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=brepnet_collate_fn,
            batch_size=self.opts.batch_size,
            num_workers=self.opts.num_workers
        )


    def test_dataloader(self):
        val_or_test = "test_set"

        # Do we want to evaluate the model using the 
        # validation set of the held out test set?
        if self.opts.test_with_validation_set is not None:
            if self.opts.test_with_validation_set:
                val_or_test = "validation_set"

        if self.opts.use_old_dataloader:
            # Legacy dataloader for json data extracted with 
            # proprietary code  
            dataset = BRepNetDatasetOld(self.opts, val_or_test)
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=None  # Solids are organized into fixed batches
            )

        dataset = BRepNetDataset(self.opts, val_or_test)
        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=brepnet_collate_fn,
            batch_size=self.opts.batch_size,
            num_workers=self.opts.num_workers
        )


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.opts.learning_rate)