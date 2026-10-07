# OpenAI image assistant

Open **OpenAI ช่วยวิเคราะห์และเตรียมภาพ** under AI settings.

1. **วิเคราะห์ภาพด้วย OpenAI** sends the reference upload to the Responses API.
   The report separates visible details, preservation constraints and hidden-part hypotheses.
   The report itself does not alter 3D model weights or the existing mesh.
2. **OpenAI สร้างภาพ 4 มุม (มีค่า API)** sends the source and report constraints to the
   Images edit API for a front/left/right/back sheet. If the upload was recognized
   as a frontal view, its original pixels are retained in the front slot.
3. The app displays all four images. Hunyuan selections switch to Hunyuan3D-2mv;
   TRELLIS stays on TRELLIS. Inspect the predictions,
   then click **สร้างโมเดล 3D**. TRELLIS receives the images through `run_multi_image`.
   Hunyuan3D-2mv receives all four images in its shape pipeline's image dictionary;
   color is projected from each reference, rather than falsely claiming a 4K AI texture.
   Single-image engines can use the analysis, but cannot consume a text anatomy plan directly.

Hidden surfaces are guesses. Contradictory views can worsen geometry, and no 100%
similarity guarantee is made. The comparison DOCX shows older Hunyuan results,
not the newly integrated TRELLIS result.

`OPENAI_API_KEY` is read only from the server environment. It is never returned to
the browser or saved in an artifact. Default vision model: `gpt-4.1`. Default image
model: `gpt-image-2.5-sunburst`, medium quality, 1536x1024, one sheet. Configure
`OPENAI_VISION_MODEL` and `OPENAI_IMAGE_MODEL` if needed. This is a cloud API with
account charges, separate from the app's offline 3D inference.

No automatic external calls occur when a page opens or a 3D job starts. Requests
have no automatic retries, and completed reference sheets are cached. Responses
use `store=false`; local reports and images are under `openai-assistance/<id>`.
API error responses do not echo credentials or submitted images.

Validation: image limits, schema request, incomplete/refusal handling, safe errors,
front preservation, view ordering and cached results using mocked API responses.
Credential availability was checked by a read-only model lookup (HTTP 200).
Live image analysis and generation were NOT executed: automatic approval review
requires explicit permission to upload the document image and incur API charges.
Multi-image TRELLIS remains to be tested with approved generated views.

Hunyuan multiview and TRELLIS multi-image cloud-assisted results still need a real GPU run.

Live analysis diagnostic returned 429 credit_balance_exhausted. The app now shows a persistent inline explanation and Billing link. Global notifications use fourview-notice to avoid HeroUI toast CSS collisions. No analysis or generated views were completed during that diagnostic. The configured key matches the suffix shown in the user's screenshot.

