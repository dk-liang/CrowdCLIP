import os.path as osp

import torch
import torch.nn as nn
import torchvision.models as models
from clip import clip

from crowdclip.utils import get_logger


from .builder import MODELS
import time


logger = get_logger(__name__)
crowd_count = ['20', '55', '90', '125', '160', '195']             #283
person_exist = ['crowd', 'tree', 'car', 'traffic light']
person_exist2 = ['human heads', 'hands', 'legs', 'road', 'house', 'shadow of man']

device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess = clip.load('ViT-B/16', device)


@MODELS.register_module()
class CrowdCLIP(nn.Module):
    def __init__(
            self,
            text_encoder_name,
            image_encoder_name,
            **kwargs,
    ) -> None:
        super().__init__()
        # print('text_encoder_name:', text_encoder_name, 'image_encoder_name:', image_encoder_name)
        if kwargs:
            logger.info(f"irrelevant kwargs: {kwargs}")

        clip_model = load_clip_to_cpu(
            text_encoder_name,
            image_encoder_name,
            root=osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", ".cache", "clip"),
        )

        # convert to float32
        clip_model.float()
        logger.info("convert `clip_model` to float32. if need fp16 model, call `clip.model.convert_weights`")

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        # self.clip_model = clip_model

        self.embed_dims = clip_model.text_projection.shape[1]

    def forward(self, images, phase='train'):
        if phase == 'train':
            text_inputs = torch.cat([clip.tokenize(f"There are {c} persons in the crowd") for c in crowd_count]).cuda()
            text_features = self.text_encoder(text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            image_features = self.image_encoder(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            count = []
            # logits2 = []
            logits = (100.0 * image_features @ text_features.T)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            for i in range(image_features.size()[0]):
                values, indices = similarity[i].topk(5)
                count.append(int(crowd_count[indices[0]]))
                # logits2.append(similarity[i].topk(1))
        else:
            new_x = images
            j = 0
            text1_inputs = torch.cat([clip.tokenize(f"The object is {p}") for p in person_exist]).cuda()

            text2_inputs = torch.cat([clip.tokenize(f"The objects are {p}") for p in person_exist2]).cuda()

            text3_inputs = torch.cat(
                [clip.tokenize(f"There are {c} persons in the crowd") for c in crowd_count]).cuda()

            text_inputs = torch.cat([text1_inputs, text2_inputs, text3_inputs])

            with torch.no_grad():
                text_features = self.text_encoder(text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            text_features_split = torch.split(text_features,
                                              [len(person_exist), len(person_exist2), len(crowd_count)])
            text_features1 = text_features_split[0]

            text_features2 = text_features_split[1]

            text_features3 = text_features_split[2]


            with torch.no_grad():
                image_features = model_clip.encode_image(new_x)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            for i in range(images.size()[0]):

                similarity = (100.0 * image_features[i].unsqueeze(0) @ text_features1.T).softmax(dim=-1)
                values, indices = similarity[0].topk(4)
                if person_exist[indices[0]] == 'crowd':

                    similarity = (100.0 * image_features[i].unsqueeze(0) @ text_features2.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(3)
                    if (person_exist2[indices[0]] == 'human heads' or person_exist2[indices[1]] == 'human heads' or
                            person_exist2[indices[2]] == 'human heads'):
                        # print('i:', i)
                        j += 1
                        if j == 1:
                            new_x = images[i].unsqueeze(0).to(device)
                        else:
                            new_x = torch.cat((new_x, images[i].unsqueeze(0).to(device)), dim=0)


            with torch.no_grad():
                image_features = self.image_encoder(new_x)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            count = []
            # logits2 = []
            logits = (100.0 * image_features @ text_features3.T)
            similarity = (100.0 * image_features @ text_features3.T).softmax(dim=-1)

            for i in range(image_features.size()[0]):
                values, indices = similarity[i].topk(5)
                count.append(int(crowd_count[indices[0]]))
                # logits2.append(similarity[i].topk(1))



        return logits, count, image_features, text_features

    def forward_text_only(self):
        text_features = self.text_encoder(sentence_embeds, psudo_sentence_tokens)

        return text_features

    def encode_image(self, x):
        return self.image_encoder(x)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.token_embedding = clip_model.token_embedding


    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    @property
    def dtype(self):
        return self.transformer.resblocks[0].mlp.c_fc.weight.dtype


def load_clip_to_cpu(
        text_encoder_name,
        image_encoder_name,
        root=osp.join(osp.expanduser("~/.cache/clip")),
):
    # text backbone
    if logger is not None:
        print_func = logger.info
    else:
        print_func = print

    print_func("Building CLIP model...")
    text_backbone_name = text_encoder_name
    print_func(f"Text backbone : {text_backbone_name}'s counterpart.")
    url = clip._MODELS[text_backbone_name]
    # print('url:', url)
    model_path = clip._download(url, root=root)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    # image backbone
    embed_dim = model.text_projection.shape[1]
    input_resolution = model.visual.input_resolution
    image_backbone_name = image_encoder_name
    print_func(f"Image backbone: {image_backbone_name}")

    if image_backbone_name != text_backbone_name:
        # remove the stochastic back-prop in vgg and alexnet
        MODEL = getattr(image_encoders, image_backbone_name, None)
        if MODEL is None:
            MODEL = getattr(models, image_backbone_name, None)
            logger.warning(f"Try PyTorch Official image model: {image_backbone_name}")
        else:
            logger.info(f"Try Custom image model: {image_backbone_name}")
        if MODEL is None:
            raise ValueError(f"Invalid torchvison model name: {image_backbone_name}")
        model.visual = MODEL(num_classes=embed_dim)
        model.visual.input_resolution = input_resolution
    else:
        print_func(f"CLIP Image encoder: {image_backbone_name}!")

    return model
