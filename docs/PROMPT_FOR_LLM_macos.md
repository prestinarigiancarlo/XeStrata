ISTRUZIONE: Esegui adesso, in modo completo, tutti e sei i compiti descritti nel documento qui sotto. Non riassumere, non chiedere conferma, non tradurre: produci direttamente gli artefatti (shader, C++, CMake, reference CPU, API vulkan.hpp, tabelle, script, checklist). Inizia dal Compito 1.

## Ruolo
Agisci come ingegnere esperto di Vulkan Compute, GLSL/SPIR-V, SYCL e GPU AMD RDNA 3. Il tuo compito è produrre
artefatti **statici** per il porting Vulkan di XeStrata che possano essere **generati, compilati e validati su macOS**,
senza accesso alla GPU AMD Radeon 780M (`gfx1103`) né al runtime RADV. Tutto ciò che richiede l'hardware va
esplicitamente differito e marcato come "DA VERIFICARE SU HARDWARE (Beelink SER8 + RADV)".

## Contesto (fatti verificati, non rispiegarli)
XeStrata è un motore di inferenza MoE interamente in SYCL. Non ha backend Vulkan. Il livello di controllo è
`strata::gpu::*` (`src/core/gpu.cpp`, `device.cpp`, `graph.cpp`, `vmm.cpp`); i kernel risolvono la coda via
`core::Runtime::get().stream(stream)` (es. `bf16_gemv.cpp:38`). I kernel WMMA AMD usano
`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` in `src/kernels/xe/amd_wmma.hpp`; `row_of()` è specifica per gfx11
(`2*e + hi`) e va ridefinita per GLSL. Le GEMM dense passano per oneMath (SYCL-only). `ggml-vulkan` è già compilato
per `tools/vision` e le dipendenze Vulkan (`glslc`, `spirv-headers`, `mesa-vulkan-drivers`) sono nel `setup.py` base.
Il graph capture è usato in decode (`plain_model.cpp:980`) e non ha equivalente 1:1 in Vulkan. La memoria host mappata
si basa sull'identità host/device pointer SYCL, non garantita in Vulkan. Subgroup a 32 lane; flag di precisione
`-foffload-fp32-prec-div -foffload-fp32-prec-sqrt`.

## Vincolo chiave di questa fase
Produci **solo** lavoro eseguibile/validabile su macOS. In particolare:

### In scope (macOS)
- Scrivere shader GLSL compute e compilarli con `glslc` (Vulkan SDK LunarG per macOS) in SPIR-V.
- Validare il SPIR-V con `spirv-val`.
- Scrivere un harness C++ Vulkan minimale compilabile su macOS contro gli header Vulkan e il loader.
- Eseguire l'harness in **smoke test** su **lavapipe** (Mesa software Vulkan) e/o **MoltenVK** per validare la
  parte API generica (device, buffer, dispatch, fence, readback).
- Scrivere una **reference CPU in C++** dell'algoritmo WMMA (porting della logica `row_of`/lane mapping di
  `amd_wmma.hpp`) per il confronto bit-exact dell'algoritmo, indipendente da SYCL.
- Progettare l'API pubblica di `vulkan.hpp` e scrivere unit test compilabili.
- Analisi statica del repository e mappatura kernel → shader.
- Piani di test, script di confronto, checklist.

### Fuori scope (differito alla Beelink SER8)
- Verifica della disponibilità effettiva di `GLSL_EXT_subgroup` / subgroup matrix su RADV.
- Esecuzione bit-exact contro il kernel SYCL `amd_wmma` (SYCL/ROCm non disponibile su macOS).
- `VK_AMD_device_host_access`, identità host/device pointer reale, graph capture su RADV.
- Qualsiasi misura di performance (prefill/decode token/s, latenza, memoria).
- Qualsiasi dichiarazione su cosa RADV supporta o su come si comporta: **non inventare e non assumere**.
  Usa "DA VERIFICARE SU HARDWARE" e indica il comando/query esatto da eseguire (es. `vulkaninfo --summary`,
  sezione subgroup di `vulkaninfo`).

## Strumenti macOS a cui fare riferimento
- `glslc` (Vulkan SDK): `glslc -fshader-stage=compute --target-env=vulkan1.3 shader.comp -o shader.spv`
- `spirv-val shader.spv`
- `spirv-dis` per ispezione
- lavapipe (`VK_ICD_FILENAMES` verso `lvp_icd.*.json`) per smoke test software
- MoltenVK per smoke test API su Metal (solo parte generica)
- CMake + un compilatore C++17 per l'harness
- `glslangValidator` come alternativa a `glslc`

## Compiti

### Compito 1 — Shader GLSL: variante subgroup-matrix e variante FMA portabile
Per la moltiplicazione 16×16×16 (f16 e i8), produci **due** shader:
- `wmma_subgroup.comp`: usa la subgroup matrix (`layout(num_rows=16, num_columns=16, nesting=32)`), con
  `#extension` opportunamente guardata. Ridefinisci esplicitamente il lane mapping gfx11 (`row_of = 2*e + hi`) e
  spiega come si traduce nel layout GLSL. Questo shader potrebbe **non compilare** con `glslc`/lavapipe: gestisci
  il caso e documenta l'errore atteso.
- `wmma_fma.comp`: fallback portabile che implementa la stessa matematica con FMA vettoriali sulle 32 lane, senza
  subgroup matrix. **Questo deve compilare e girare su lavapipe**, così possiamo validare l'algoritmo e il lane
  mapping già su macOS.
Per entrambi: dichiara i binding (SSBO input/output/scala), le dimensioni, e i vincoli di allineamento. Fornisci i
comandi esatti di compilazione e validazione, e l'output atteso di `spirv-val`.

### Compito 2 — Harness Vulkan C++ minimale (compilabile su macOS)
Scrivi un harness C++17 (`harness/main.cpp` + `CMakeLists.txt`) che:
- crea istanza/device, seleziona una GPU (con override via env per forzare lavapipe/MoltenVK),
- alloca buffer host-visible, carica input, dispatcha lo shader, legge l'output,
- confronta con la reference CPU (Compito 3) e stampa un esito bit-exact per elemento,
- ha un path "RADV stub": se il device non è RADV, salta i test hardware-specifici e stampa "DA VERIFICARE SU
  HARDWARE".
Includi le istruzioni per eseguirlo su lavapipe e su MoltenVK. Specifica chiaramente quali asserzioni sono
significative su macOS (API generica + fallback FMA) e quali no (subgroup matrix).

### Compito 3 — Reference CPU bit-exact dell'algoritmo WMMA
Scrivi una reference C++ che replichi la matematica e il lane mapping di `amd_wmma.hpp` (gfx11), indipendente da
SYCL e da Vulkan. Questa è la verità di riferimento su macOS per il fallback FMA. Documenta come, in seguito sulla
Beelink, la stessa reference verrà confrontata con l'output SYCL per chiudere il ciclo (test di parità SYCL ↔
reference ↔ GLSL).

### Compito 4 — Design dell'API `vulkan.hpp`
Definisci l'API pubblica che replica il contratto `strata::gpu::*` (stream, event, alloc_device, alloc_host,
copy, memset, wait, graph capture) con firme, struct e semantica di errore (`false`/null, `last_error()`).
Scrivi unit test compilabili su macOS per la parte pura (strutture, mapping handle, stato) e mock per la parte
driver. Evidenzia i punti che richiedono una decisione sulla macchina fisica (identità host/device pointer,
formato dei semafori/fence, graph capture) e proponi 2-3 opzioni con criteri di scelta misurabili.

### Compito 5 — Mappatura kernel → shader e piano di porting incrementale
Dai 41 file con codice di lancio in `src/kernels/xe/`, produci una tabella: file, operazione, primitive SYCL usate,
complessità di traduzione GLSL (bassa/media/alta), dipendenza da MMA, test di parità esistente
(`*_parity.cpp`). Ordina i gruppi per rischio crescente e indica quali possono essere validati interamente su macOS
(primitivi senza MMA) e quali richiedono la Beelink. Non stimare percentuali o tempi inventati: usa "bassa/media/
alta" e motiva.

### Compito 6 — Script di confronto e checklist hardware
- Script (`compare.py` o shell) che confronta due file binari float/int e riporta differenze bit-exact, max ULP,
  NaN/Inf.
- Checklist di verifica da eseguire **sulla Beelink**: `vulkaninfo --summary`, sezione subgroup, presenza di
  `VK_AMD_device_host_access`, test di identità host/device pointer, prova di graph capture, confronto bit-exact
  contro SYCL, misure di performance. Per ogni voce: comando esatto, output atteso, criterio go/no-go.

## Formato della risposta
Markdown, sezioni 1–6. Blocchi di codice per shader, C++, CMake, comandi shell. Ogni affermazione che dipende da
RADV/gfx1103 va marcata **DA VERIFICARE SU HARDWARE** con il comando/query che la chiuderà. Non dichiarare
verifiche hardware, non inventare risultati di performance, non assumere estensioni RADV. Se un artefatto non è
compilabile con gli strumenti macOS indicati, spiegalo e fornisci l'alternativa portabile.
