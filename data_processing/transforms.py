from torchvision import transforms

def build_clip_eval_transform(image_size=224):
    """Deterministic, label-agnostic transform for validation/test."""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])

def build_clip_train_transform(image_size=224):
    """Train-time only random augmentation. Do NOT use on val/test."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])
