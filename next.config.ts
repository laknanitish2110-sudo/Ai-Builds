import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  webpack: (config, { isServer }) => {
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      encoding: false,
      path: false,
      crypto: false,
    };

    if (!isServer) {
      config.resolve.alias = {
        ...config.resolve.alias,
        "onnxruntime-node": false,
      };
    }

    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
    };

    return config;
  },
};

export default nextConfig;
