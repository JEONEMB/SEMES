import fs from 'node:fs/promises';
import {finalizePresentation} from 'file:///C:/Users/SSAFY/.codex/plugins/cache/openai-primary-runtime/presentations/26.1024.11018/skills/presentations/container_tools/artifact_tool_utils.mjs';
const base='C:/Users/SSAFY/Desktop/hwlee/projects/2026-10_SEMES/S15P31A507/docs/hye';
const skill='C:/Users/SSAFY/.codex/plugins/cache/openai-primary-runtime/presentations/26.1024.11018/skills/presentations';
const r=await finalizePresentation({workspaceDir:base,candidatePath:base+'/ppt_v1_work/candidate.pptx',finalPath:base+'/ppt_v1_work/deliverable/2026-10-29_중간_v1.pptx',pythonExecutable:'C:/Users/SSAFY/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[4,6,8,11,13].flatMap(n=>['--require-native-table-slide',String(n)])],explicitTotalSlideCount:14,requiredNativeTableOwnerSlides:[4,6,8,11,13],fontPolicy:{basis:'reference',families:['Noto Sans KR'],referencePath:base+'/2026-10-29_중간.pptx',referenceSha256:'483b319e07cf268f1dbf02d4f50c6a6420799c234bddd028233bd9cd6f9c1f30'},verifyArtifactToolImport:true,receiptPath:base+'/ppt_v1_work/validation.json'});console.log(JSON.stringify(r));



