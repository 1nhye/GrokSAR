_base_ = [
    '../retinanet/retinanet_r50_fpn_6x_AIR-SARShip-1.0.py',
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

model = dict(
    backbone=dict(
        _delete_=True,  # 기존 백본 제거
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2),  # FPN에 사용할 백본 출력
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        _delete_=True,  # 기존 FPN 제거
        type='groksar.FrequencySpatialFPN',  # 새로운 neck 정의
        in_channels=[192, 384, 768],  # Swin Transformer 출력 채널
        out_channels=256,
        start_level=0,
        num_outs=5,
        add_extra_convs='on_input',  # 추가 계층 생성
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)
    )
)
