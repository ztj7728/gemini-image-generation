import { GoogleGenAI } from "@google/genai";
import fs from "node:fs/promises";
import path from "node:path";

function parseArgs(argv) {
  const parsed = {};

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];

    if (!token.startsWith("--")) {
      continue;
    }

    const body = token.slice(2);
    const separatorIndex = body.indexOf("=");

    if (separatorIndex >= 0) {
      const key = body.slice(0, separatorIndex);
      const value = body.slice(separatorIndex + 1);
      parsed[key] = value;
      continue;
    }

    const nextToken = argv[index + 1];
    if (!nextToken || nextToken.startsWith("--")) {
      parsed[body] = "true";
      continue;
    }

    parsed[body] = nextToken;
    index += 1;
  }

  return parsed;
}

function requireArg(value, name) {
  if (!value) {
    throw new Error(`Missing required argument: --${name}`);
  }

  return value;
}

function readImageConfigArgs(args) {
  const aspectRatio = args.aspectRatio || args["aspect-ratio"];
  const imageSize = args.imageSize || args["image-size"];

  if (!aspectRatio && !imageSize) {
    return undefined;
  }

  return {
    ...(aspectRatio ? { aspectRatio } : {}),
    ...(imageSize ? { imageSize } : {}),
  };
}

function extensionForMimeType(mimeType) {
  switch (mimeType) {
    case "image/png":
      return ".png";
    case "image/jpeg":
      return ".jpg";
    case "image/webp":
      return ".webp";
    default:
      return ".bin";
  }
}

function resolveOutputPath(outputPath, imageIndex, mimeType) {
  const parsedPath = path.parse(outputPath);
  const fallbackExtension = extensionForMimeType(mimeType);
  const extension = parsedPath.ext || fallbackExtension;
  const baseName = parsedPath.ext ? parsedPath.name : parsedPath.base;
  const imageSuffix = imageIndex === 0 ? "" : `-${imageIndex + 1}`;

  return path.join(parsedPath.dir, `${baseName}${imageSuffix}${extension}`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const prompt = requireArg(args.prompt, "prompt");
  const output = requireArg(args.output, "output");
  const imageConfig = readImageConfigArgs(args);
  const apiKey = process.env.GEMINI_API_KEY;
  const model = process.env.GEMINI_MODEL_ID;
  const baseUrl = process.env.GEMINI_BASE_URL?.replace(/\/$/, "");

  if (!apiKey) {
    throw new Error("Missing GEMINI_API_KEY in the environment");
  }

  if (!model) {
    throw new Error("Missing GEMINI_MODEL_ID in the environment");
  }

  const ai = new GoogleGenAI({
    apiKey,
    httpOptions: baseUrl ? { baseUrl } : undefined
  });
  const response = await ai.models.generateContent({
    model,
    contents: prompt,
    ...(imageConfig
      ? {
          config: {
            imageConfig,
          },
        }
      : {}),
  });

  const parts = response?.candidates?.[0]?.content?.parts || [];

  if (parts.length === 0) {
    throw new Error("Gemini returned no content parts");
  }

  const savedImages = [];

  for (const part of parts) {
    if (part.text) {
      console.log(`TEXT: ${part.text}`);
    }

    if (!part.inlineData?.data) {
      continue;
    }

    const filePath = resolveOutputPath(output, savedImages.length, part.inlineData.mimeType);
    const imageBuffer = Buffer.from(part.inlineData.data, "base64");

    await fs.mkdir(path.dirname(filePath), { recursive: true });
    await fs.writeFile(filePath, imageBuffer);

    savedImages.push(filePath);
    console.log(`IMAGE: ${filePath}`);
  }

  if (savedImages.length === 0) {
    throw new Error("Gemini returned text but no image data");
  }

  console.log(
    JSON.stringify(
      {
        model,
        prompt,
        ...(imageConfig ? { imageConfig } : {}),
        images: savedImages,
      },
      null,
      2,
    ),
  );
}

main().catch((error) => {
  console.error(`ERROR: ${error.message}`);
  process.exitCode = 1;
});
