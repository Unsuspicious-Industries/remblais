{
  description = "Remblais: local image morphing with optimal transport";

  inputs = {
    nixpkgs.url     = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        libs = with pkgs; [
          stdenv.cc.cc.lib
          zlib
          libffi
        ];
      in {
        devShells.default = pkgs.mkShell {
          packages = [ pkgs.python313 pkgs.uv pkgs.clang ] ++ libs;

          env = {
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath libs;
            UV_PYTHON = "${pkgs.python313}/bin/python3";
          };

          shellHook = ''
            export CC="${pkgs.clang}/bin/clang"
          '';
        };
      }
    );
}
