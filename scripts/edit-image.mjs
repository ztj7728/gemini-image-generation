import { GoogleGenAI } from "@google/genai";
import fs from "node:fs/promises";
import path from "node:path";

function appendArgValue(parsed, key, value) {
  if (!(key in parsed)) {
    parsed[key] = value;
    return;
  }

  if (Array.isArray(parsed[key])) {
    parsed[key].push(value);
    return;
  }

  parsed[key] = [parsed[key], value];
}

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
      appendArgValue(parsed, key, value);
      continue;
    }

    const nextToken = argv[index + 1];
    if (!nextToken || nextToken.startsWith("--")) {
      appendArgValue(parsed, body, "true");
      continue;
    }

    appendArgValue(parsed, body, nextToken);
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

function normalizeArgList(value) {
  if (value === undefined) {
    return [];
  }

  const values = Array.isArray(value) ? value : [value];

  return values
    .flatMap((entry) => entry.split(","))
    .map((entry) => entry.trim())
    .filter(Boolean);
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

function mimeTypeForPath(filePath) {
  const extension = path.extname(filePath).toLowerCase();

  switch (extension) {
    case ".png":
      return "image/png";
    case ".jpg":
    case ".jpeg":
      return "image/jpeg";
    case ".webp":
      return "image/webp";
    default:
      throw new Error(
        `Unable to infer mime type from ${filePath}. Pass --mime-type explicitly.`,
      );
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
  const inputs = normalizeArgList(requireArg(args.input, "input"));
  const output = requireArg(args.output, "output");
  const mimeTypes = normalizeArgList(args["mime-type"]);
  const imageConfig = readImageConfigArgs(args);
  const apiKey = process.env.GEMINI_API_KEY;
  const model = process.env.GEMINI_MODEL_ID;
  const baseUrl = process.env.GEMINI_BASE_URL?.replace(/\/$/, "");

  if (inputs.length === 0) {
    throw new Error("At least one --input value is required");
  }

  if (mimeTypes.length > 1 && mimeTypes.length !== inputs.length) {
    throw new Error("Pass either one --mime-type for all inputs or one --mime-type per --input");
  }

  if (!apiKey) {
    throw new Error("Missing GEMINI_API_KEY in the environment");
  }

  if (!model) {
    throw new Error("Missing GEMINI_MODEL_ID in the environment");
  }

  const inputContents = await Promise.all(
    inputs.map(async (input, index) => {
      const imageBuffer = await fs.readFile(input);
      const mimeType = mimeTypes[index] || mimeTypes[0] || mimeTypeForPath(input);

      return {
        input,
        mimeType,
        inlineData: {
          mimeType,
          data: imageBuffer.toString("base64"),
        },
      };
    }),
  );
  const ai = new GoogleGenAI({
    apiKey,
    httpOptions: baseUrl ? { baseUrl } : undefined,
  });
  const response = await ai.models.generateContent({
    model,
    contents: [{ text: prompt }, ...inputContents.map((entry) => entry.inlineData)],
    config: {
      responseModalities: ["TEXT", "IMAGE"],
      ...(imageConfig ? { imageConfig } : {}),
    },
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
    const generatedImageBuffer = Buffer.from(part.inlineData.data, "base64");

    await fs.mkdir(path.dirname(filePath), { recursive: true });
    await fs.writeFile(filePath, generatedImageBuffer);

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
        inputs: inputContents.map(({ input, mimeType }) => ({ input, mimeType })),
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