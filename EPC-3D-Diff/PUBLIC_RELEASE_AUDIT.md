# Public-release audit

This file records issues identified while preparing the uploaded EPC-3D-Diff code for a public GitHub release. These items should be resolved and regression-tested before publishing the repository as the implementation corresponding to the paper.

## Release blockers

1. **Missing physics helper modules.** `train.py` imports `src.misc`, `ops.astracd_op`, `ops.astra_autograd`, and `ops.Elekta_ASTRA_Geom`, and also expects `ops/Elekta_geometry.xml`. These files were not included in the current upload.

2. **Projection-equivariance gradient path requires verification.** In the uploaded training script, the complete projection-equivariance calculation is wrapped in `torch.no_grad()`. That makes the returned equivariance term non-differentiable with respect to the synthesized CT in ordinary PyTorch autograd. The paper describes this term as part of the training objective, so the public implementation should preserve gradients through the synthesized-CT projection path while detaching only the reference projection.

3. **Paper/code loss-weight mismatch.** The uploaded script forms the image/diffusion main loss as:
   `0.6 * diffusion + 0.2 * L1 + 0.2 * edge + 0.1 * Laplacian`.
   The paper describes the total objective as `L_DDPM + λ1 L1 + λ2 Ledge + λ3 Llap + λeq Leq` and reports `λ1=0.6, λ2=0.2, λ3=0.2, λeq=0.1`. The authors should confirm the exact coefficients used for the reported experiments before the code is changed.

4. **Number of equivariance rotations requires confirmation.** The paper reports two random rotations per sample, whereas the uploaded entry-point configuration currently sets one rotation.

5. **Mixed-domain balanced batching bug.** `build_dataset("JUST_NWH")` returns a `ConcatDataset`, but the uploaded entry point checks `isinstance(train_dataset, tuple)`, so the balanced sampler branch is not entered. The sampler argument order should also follow the actual `ConcatDataset` ordering.

6. **Optional raw projection data-consistency loss is not part of Eq. (10).** The uploaded training script contains a separate `rawDC_loss` term in addition to the projection-equivariance term. This is not listed in the paper's total objective and should either be removed from the paper-matching training path or explicitly documented as an optional experimental extension.

7. **JUST raw-data path in `rawDC_loss`.** The current implementation references NWH MAT-file contents for raw projection consistency; the JUST branch does not define the corresponding raw-projection variable. Mixed-domain behavior therefore needs explicit handling if `rawDC_loss` is retained.

8. **Autoencoder pretraining batch unpacking.** The 3D dataset returns four values `(cbct, ct, patient_id, domain_id)`, while the current pretraining loop expects three. This should be corrected before enabling AE pretraining from the public entry point.

## Public-code cleanup still recommended

- replace hard-coded local paths and GPU indices with CLI/YAML configuration;
- remove local debugging comments, emojis, and experiment-specific resume values;
- centralize the latent encoder/decoder so training and testing import one shared implementation;
- add deterministic seeding and reproducibility notes;
- pin exact package versions from the environment used for the paper;
- add a small synthetic-data smoke test that does not require patient data;
- verify that no patient data, identifiers, raw projections, checkpoints, or local paths are committed;
- run formatting/linting and a clean-environment installation test.

## Licensing

The source headers use the Mozilla Public License 2.0 notice requested by the authors. Before release, the authors should confirm that the stated copyright ownership is consistent with University of Dundee/JUST/other institutional IP policies and that any third-party ASTRA/TIGRE-derived code is distributed under compatible terms.
