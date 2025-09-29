
import { GoogleGenAI, Modality } from "@google/genai";
import type { GeneratedContent } from '../types';

if (!process.env.API_KEY) {
  throw new Error("API_KEY environment variable is not set.");
}

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

export async function editImage(
    prompt: string,
    imageParts: { base64: string; mimeType: string }[],
    maskBase64: string | null
): Promise<GeneratedContent> {
  try {
    let fullPrompt = prompt;
    const parts: any[] = [];

    // The primary image is always the first one.
    if (imageParts.length > 0) {
        parts.push({
            inlineData: { data: imageParts[0].base64, mimeType: imageParts[0].mimeType },
        });
    }

    if (maskBase64) {
      parts.push({
        inlineData: { data: maskBase64, mimeType: 'image/png' },
      });
      fullPrompt = `Apply the following instruction only to the masked area of the image: "${prompt}". Preserve the unmasked area.`;
    }
    
    // Add any remaining images (secondary, tertiary, etc.)
    if (imageParts.length > 1) {
        imageParts.slice(1).forEach(img => {
            parts.push({
                inlineData: { data: img.base64, mimeType: img.mimeType },
            });
        });
    }

    parts.push({ text: fullPrompt });


    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash-image-preview',
      contents: { parts },
      config: {
        responseModalities: [Modality.IMAGE, Modality.TEXT],
      },
    });

    const result: GeneratedContent = { imageUrl: null, text: null };
    const responseParts = response.candidates?.[0]?.content?.parts;

    if (responseParts) {
      for (const part of responseParts) {
        if (part.text) {
          result.text = (result.text ? result.text + "\n" : "") + part.text;
        } else if (part.inlineData) {
          result.imageUrl = `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`;
        }
      }
    }

    if (!result.imageUrl) {
        let errorMessage;
        if (result.text) {
            errorMessage = `The model responded: "${result.text}"`;
        } else {
            const finishReason = response.candidates?.[0]?.finishReason;
            const safetyRatings = response.candidates?.[0]?.safetyRatings;
            errorMessage = "The model did not return an image. It might have refused the request. Please try a different image or prompt.";
            
            if (finishReason === 'SAFETY') {
                const blockedCategories = safetyRatings?.filter(r => r.blocked).map(r => r.category).join(', ');
                errorMessage = `The request was blocked for safety reasons. Categories: ${blockedCategories || 'Unknown'}. Please modify your prompt or image.`;
            }
        }
        throw new Error(errorMessage);
    }

    return result;

  } catch (error) {
    console.error("Error calling Gemini API:", error);
    if (error instanceof Error) {
        let errorMessage = error.message;
        try {
            const parsedError = JSON.parse(errorMessage);
            if (parsedError.error && parsedError.error.message) {
                if (parsedError.error.status === 'RESOURCE_EXHAUSTED') {
                    errorMessage = "You've likely exceeded the request limit. Please wait a moment before trying again.";
                } else if (parsedError.error.code === 500 || parsedError.error.status === 'UNKNOWN') {
                    errorMessage = "An unexpected server error occurred. This might be a temporary issue. Please try again in a few moments.";
                } else {
                    errorMessage = parsedError.error.message;
                }
            }
        } catch (e) {}
        throw new Error(errorMessage);
    }
    throw new Error("An unknown error occurred while communicating with the API.");
  }
}

export async function generateImageEditsBatch(
    prompt: string,
    imageParts: { base64: string; mimeType: string }[]
): Promise<string[]> {
    try {
        const promises: Promise<GeneratedContent>[] = [];
        for (let i = 0; i < 4; i++) {
            // Pass null for maskBase64 as this flow doesn't use it.
            promises.push(editImage(prompt, imageParts, null));
        }
        const results = await Promise.all(promises);
        const imageUrls = results.map(r => r.imageUrl).filter((url): url is string => !!url);
        
        if (imageUrls.length === 0) {
          throw new Error("Failed to generate any image variations. The model may have refused the request.");
        }
        
        return imageUrls;
    } catch (error) {
        console.error("Error generating image edits batch:", error);
        if (error instanceof Error) {
            // Re-throw the specific error message from a failed child `editImage` call
            throw new Error(error.message);
        }
        throw new Error("An unknown error occurred during batch image generation.");
    }
}

type ImageAspectRatio = '1:1' | '16:9' | '9:16' | '4:3' | '3:4';

export async function generateImageFromText(
    prompt: string,
    aspectRatio: ImageAspectRatio
): Promise<GeneratedContent> {
  try {
    const response = await ai.models.generateImages({
        model: 'imagen-4.0-generate-001',
        prompt: prompt,
        config: {
          numberOfImages: 1,
          outputMimeType: 'image/png',
          aspectRatio: aspectRatio,
        },
    });

    if (!response.generatedImages || response.generatedImages.length === 0) {
        throw new Error("The model did not return an image. It might have refused the request.");
    }

    const base64ImageBytes: string = response.generatedImages[0].image.imageBytes;
    const imageUrl = `data:image/png;base64,${base64ImageBytes}`;

    return { imageUrl, text: null };

  } catch (error) {
    console.error("Error calling Gemini API for text-to-image:", error);
    if (error instanceof Error) {
        let errorMessage = error.message;
        try {
            const parsedError = JSON.parse(errorMessage);
            if (parsedError.error && parsedError.error.message) {
                if (parsedError.error.status === 'RESOURCE_EXHAUSTED') {
                    errorMessage = "You've likely exceeded the request limit. Please wait a moment before trying again.";
                } else if (parsedError.error.code === 500 || parsedError.error.status === 'UNKNOWN') {
                    errorMessage = "An unexpected server error occurred. This might be a temporary issue. Please try again in a few moments.";
                } else {
                    errorMessage = parsedError.error.message;
                }
            }
        } catch (e) {}
        throw new Error(errorMessage);
    }
    throw new Error("An unknown error occurred while communicating with the API.");
  }
}

export async function generateVideo(
    prompt: string,
    image: { base64: string; mimeType: string } | null,
    aspectRatio: '16:9' | '9:16',
    onProgress: (message: string) => void
): Promise<string> {
    try {
        onProgress("Initializing video generation...");

        // FIX: The `request` object was explicitly typed as `any`, which caused a loss of type
        // information for the `operation` variable returned by `generateVideos`. This could lead
        // to a TypeScript error. By allowing TypeScript to infer the type, we ensure
        // `operation` is correctly typed, resolving the error.
        const request = {
            model: 'veo-2.0-generate-001',
            prompt: prompt,
            config: {
                numberOfVideos: 1,
                aspectRatio: aspectRatio
            },
            ...(image && {
                image: {
                    imageBytes: image.base64,
                    mimeType: image.mimeType
                }
            })
        };

        let operation = await ai.models.generateVideos(request);
        
        onProgress("Polling for results, this may take a few minutes...");

        while (!operation.done) {
            await new Promise(resolve => setTimeout(resolve, 10000));
            operation = await ai.operations.getVideosOperation({ operation: operation });
        }

        if (operation.error) {
            // FIX: The type of `operation.error.message` is `unknown`, which is not assignable to the `Error` constructor.
            // This fix ensures that only a string is passed by checking the type first and providing a fallback.
            throw new Error(typeof operation.error.message === 'string' ? (operation.error.message || "Video generation failed during operation.") : "Video generation failed during operation.");
        }

        const downloadLink = operation.response?.generatedVideos?.[0]?.video?.uri;

        if (!downloadLink) {
            throw new Error("Video generation completed, but no download link was found.");
        }

        return `${downloadLink}&key=${process.env.API_KEY}`;

    } catch (error) {
        console.error("Error calling Video Generation API:", error);
        // FIX: Replaced the error handling logic with the more robust version from the `editImage` function. 
        // This provides more specific user feedback for common API issues and resolves a potential type error.
        if (error instanceof Error) {
            let errorMessage = error.message;
            try {
                const parsedError = JSON.parse(errorMessage);
                if (parsedError.error && parsedError.error.message) {
                    if (parsedError.error.status === 'RESOURCE_EXHAUSTED') {
                        errorMessage = "You've likely exceeded the request limit. Please wait a moment before trying again.";
                    } else if (parsedError.error.code === 500 || parsedError.error.status === 'UNKNOWN') {
                        errorMessage = "An unexpected server error occurred. This might be a temporary issue. Please try again in a few moments.";
                    } else {
                        errorMessage = parsedError.error.message;
                    }
                }
            } catch (e) {}
            throw new Error(errorMessage);
        }
        throw new Error("An unknown error occurred during video generation.");
    }
}