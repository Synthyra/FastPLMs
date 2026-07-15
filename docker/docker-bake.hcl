variable "REGISTRY" {
  default = "local"
}

variable "TAG" {
  default = "dev"
}

group "default" {
  targets = ["runtime", "candidate"]
}

group "check" {
  targets = ["candidate", "candidate-structure", "candidate-fp8", "candidate-artifact"]
}

group "references" {
  targets = [
    "reference-ankh",
    "reference-biohub-esm",
    "reference-boltz2",
    "reference-dplm",
    "reference-e1",
    "reference-esm2",
    "reference-esmfold",
    "reference-esmfold2",
    "reference-protein-ttt",
  ]
}

target "common" {
  context    = "."
  dockerfile = "docker/Dockerfile"
  platforms  = ["linux/amd64"]
}

target "runtime" {
  inherits = ["common"]
  target   = "runtime"
  tags     = ["${REGISTRY}/fastplms-runtime:${TAG}"]
  args = {
    FASTPLMS_RUNTIME_PROFILE = "core"
  }
}

target "runtime-fp8" {
  inherits = ["common"]
  target   = "runtime"
  tags     = ["${REGISTRY}/fastplms-runtime-fp8:${TAG}"]
  args = {
    FASTPLMS_RUNTIME_PROFILE = "esmfold2-fp8"
  }
}

target "candidate" {
  inherits = ["common"]
  target   = "candidate"
  tags     = ["${REGISTRY}/fastplms-candidate:${TAG}"]
}

target "candidate-structure" {
  inherits = ["common"]
  target   = "candidate-structure"
  tags     = ["${REGISTRY}/fastplms-structure:${TAG}"]
}

target "candidate-fp8" {
  inherits = ["common"]
  target   = "candidate-fp8"
  tags     = ["${REGISTRY}/fastplms-fp8:${TAG}"]
}

target "candidate-artifact" {
  inherits = ["common"]
  target   = "candidate-artifact"
  tags     = ["${REGISTRY}/fastplms-artifact:${TAG}"]
}

target "reference-ankh" {
  inherits = ["common"]
  target   = "reference-ankh"
  tags     = ["${REGISTRY}/fastplms-reference-ankh:${TAG}"]
  contexts = {
    upstream_ankh = "vendor/upstream/ankh"
  }
}

target "reference-biohub-esm" {
  inherits = ["common"]
  target   = "reference-biohub-esm"
  tags     = ["${REGISTRY}/fastplms-reference-biohub-esm:${TAG}"]
  contexts = {
    upstream_biohub_esm          = "vendor/upstream/biohub-esm"
    upstream_biohub_transformers = "vendor/upstream/biohub-transformers"
  }
}

target "reference-boltz2" {
  inherits = ["common"]
  target   = "reference-boltz2"
  tags     = ["${REGISTRY}/fastplms-reference-boltz2:${TAG}"]
  contexts = {
    upstream_boltz = "vendor/upstream/boltz"
  }
}

target "reference-boltz2-same-runtime" {
  inherits = ["common"]
  target   = "reference-boltz2-same-runtime"
  tags     = ["${REGISTRY}/fastplms-reference-boltz2-same-runtime:${TAG}"]
  contexts = {
    upstream_boltz = "vendor/upstream/boltz"
  }
}

target "reference-dplm" {
  inherits = ["common"]
  target   = "reference-dplm"
  tags     = ["${REGISTRY}/fastplms-reference-dplm:${TAG}"]
  contexts = {
    upstream_dplm = "vendor/upstream/dplm"
  }
}

target "reference-e1" {
  inherits = ["common"]
  target   = "reference-e1"
  tags     = ["${REGISTRY}/fastplms-reference-e1:${TAG}"]
  contexts = {
    upstream_e1 = "vendor/upstream/e1"
  }
}

target "reference-esm2" {
  inherits = ["common"]
  target   = "reference-esm2"
  tags     = ["${REGISTRY}/fastplms-reference-esm2:${TAG}"]
  contexts = {
    upstream_fair_esm = "vendor/upstream/fair-esm"
  }
}

target "reference-esmfold" {
  inherits = ["common"]
  target   = "reference-esmfold"
  tags     = ["${REGISTRY}/fastplms-reference-esmfold:${TAG}"]
  contexts = {
    upstream_fair_esm = "vendor/upstream/fair-esm"
    upstream_openfold = "vendor/upstream/openfold"
  }
}

target "reference-esmfold2" {
  inherits = ["common"]
  target   = "reference-esmfold2"
  tags     = ["${REGISTRY}/fastplms-reference-esmfold2:${TAG}"]
  contexts = {
    upstream_biohub_esm          = "vendor/upstream/biohub-esm"
    upstream_biohub_transformers = "vendor/upstream/biohub-transformers"
  }
}

target "reference-protein-ttt" {
  inherits = ["common"]
  target   = "reference-protein-ttt"
  tags     = ["${REGISTRY}/fastplms-reference-protein-ttt:${TAG}"]
  contexts = {
    upstream_protein_ttt = "vendor/upstream/protein-ttt"
  }
}
