import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedMultimodalNet(nn.Module):
    def __init__(self,
                 num_num_features,        # число числовых (табличных) фич
                 num_brands,              # число уникальных брендов
                 num_types,               # число уникальных типов
                 text_emb_dim=384,        # размер готовых текстовых эмбеддингов
                 img_emb_dim=512,         # размер готовых эмбеддингов изображений
                 proj_dim=128,            # размер проекции для каждой модальности
                 fusion_hidden=256,
                 num_classes=1,           # 1 -> бинар (BCEWithLogits), >1 -> multi-class (CrossEntropy)
                 p_drop_modality=0.4):
        super().__init__()
        self.p_drop_modality = p_drop_modality
        self.num_classes = num_classes

        # === 1) числовые признаки -> proj_dim
        self.num_proj = nn.Sequential(
            nn.Linear(num_num_features, proj_dim),
            nn.ReLU(),
            nn.BatchNorm1d(proj_dim),
            nn.Dropout(0.1)
        )

        #  === 2) Категориальные признаки 
        emb_dim_cat = proj_dim // 2  # размер эмбеддинга категориального признака
        self.brand_emb = nn.Embedding(num_brands, emb_dim_cat)
        self.com_type_emb = nn.Embedding(num_types, emb_dim_cat)

        # после конкатенации эмбеддингов -> proj_dim
        self.cat_proj = nn.Sequential(
            nn.Linear(2 * emb_dim_cat, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(0.1)
        )

        #  === 3) Текстовые эмбеддинги -> proj_dim
        self.text_proj = nn.Sequential(
            nn.Linear(text_emb_dim, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(0.1)
        )

        #  === 4)эмбеддинги изображений -> proj_dim
        # resnet
        # ...
        self.img_proj = nn.Sequential(
            nn.Linear(img_emb_dim, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(0.1)
        )

        # Гейты (по сути скалярные веса, но реализуем векторно)
        self.gate_num = nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.Sigmoid())
        self.gate_cat = nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.Sigmoid())
        self.gate_text = nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.Sigmoid())
        self.gate_img  = nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.Sigmoid())

        # Fusion + classifier
        self.fusion = nn.Sequential(
            nn.Linear(proj_dim * 4, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden // 2, num_classes)
        )

    def modality_dropout(self, x, p):
        if not self.training or p <= 0:
            return x
        if torch.rand(1).item() < p:
            return torch.zeros_like(x)
        return x

    def forward(self, num_x, brand_x, comtype_x, text_emb, img_emb, has_img=None):
        """
        num_x:     [B, num_num_features] — числовые признаки
        brand_x:   [B] индексы брендов
        comtype_x: [B] индексы типов
        text_emb:  [B, text_emb_dim] эмбеддинги текста
        img_emb:   [B, img_emb_dim] эмбеддинги картинок
        has_img:   [B] (0/1), маска наличия изображения
        """

        # 1. Числовые
        num_h = self.num_proj(num_x)

        # 2. Категориальные
        brand_h = self.brand_emb(brand_x)
        comtype_h = self.com_type_emb(comtype_x)
        cat_h = torch.cat([brand_h, comtype_h], dim=1)
        cat_h = self.cat_proj(cat_h)

        # 3. Текст
        text_h = self.text_proj(text_emb)

        # 4. Изображения (+ маска)
        img_h = self.img_proj(img_emb)
        if has_img is not None:
            mask = has_img.view(-1,1).to(img_h.dtype)
            img_h = img_h * mask
        
        # Modality dropout
        num_h  = self.modality_dropout(num_h,  self.p_drop_modality)
        cat_h  = self.modality_dropout(cat_h,  self.p_drop_modality)
        text_h = self.modality_dropout(text_h, self.p_drop_modality)
        img_h  = self.modality_dropout(img_h,  self.p_drop_modality)


        # Gating
        num_out  = self.gate_num(num_h) * num_h
        cat_out  = self.gate_cat(cat_h) * cat_h
        text_out = self.gate_text(text_h) * text_h
        img_out  = self.gate_img(img_h) * img_h

        # Fuse
        fused = torch.cat([num_out, cat_out, text_out, img_out], dim=1)  # [B, proj_dim*4]
        logits = self.fusion(fused)
        return logits