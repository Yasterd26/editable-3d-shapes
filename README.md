
---

# 🚀 Editable 3D Shapes

### *Learning to Reconstruct and Edit 3D Objects using DeepSDF*

This project explores **implicit 3D shape representations** using **Deep Signed Distance Functions (DeepSDF)** and demonstrates how **semantic edits** can be applied directly in **latent space**.

Instead of storing shapes as meshes or voxels, we learn a **continuous neural function** that defines the surface — enabling smooth reconstruction and controlled shape modification.

---

## 🧠 What This Project Does

* Trains a **DeepSDF model** on 3D shapes
* Learns a **latent vector per object**
* Reconstructs meshes using **Marching Cubes**
* Performs **part decomposition** using geometry
* Applies **semantic edits** (e.g., modifying a mug handle) via latent optimization

---

## 🏗️ Pipeline

```
GLB Mesh → SDF Sampling → DeepSDF Training
        → Latent Optimization → Mesh Reconstruction
        → Part Decomposition → Semantic Editing
```

---

## 🔬 Key Learnings

* Implicit models are **powerful but fragile**
* Reconstruction depends heavily on **SDF zero-crossing**
* Semantic edits require **careful loss balancing**
* Debugging geometry is harder than training networks

---

## ⚠️ Challenges Faced

* GLB → mesh conversion issues
* Marching cubes failures due to invalid SDF ranges
* Latent drift during editing
* Subtle edits not always surfacing geometrically

All were addressed through architectural fixes, constraints, and careful debugging.

---

## 📚 References

* **DeepSDF** – Park et al.
* **ShapeNet**
* `trimesh`, `scikit-image`

---

## 🔮 What’s Next

* Stronger semantic constraints
* Quantitative evaluation (Chamfer Distance)
* Better part segmentation
* More expressive edits

---

## 👨‍🎓 Author

**Divyansh**
Second Year Undergraduate | BYOP 2025

---
