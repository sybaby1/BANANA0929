import React, { useState, useCallback, useEffect } from 'react';
import { TRANSFORMATIONS } from './constants';
import { editImage, generateVideo, generateImageFromText, generateImageEditsBatch } from './services/geminiService';
import type { GeneratedContent, Transformation } from './types';
import TransformationSelector from './components/TransformationSelector';
import ResultDisplay from './components/ResultDisplay';
import LoadingSpinner from './components/LoadingSpinner';
import ErrorMessage from './components/ErrorMessage';
import ImageEditorCanvas from './components/ImageEditorCanvas';
import { dataUrlToFile, embedWatermark, loadImage, resizeImageToMatch, downloadImage, addVisibleWatermark } from './utils/fileUtils';
import ImagePreviewModal from './components/ImagePreviewModal';
import MultiImageUploader from './components/MultiImageUploader';
import HistoryPanel from './components/HistoryPanel';
import { useTranslation } from './i18n/context';
import LanguageSwitcher from './components/LanguageSwitcher';
import ThemeSwitcher from './components/ThemeSwitcher';
import MultiImageGridUploader from './components/MultiImageGridUploader';

type ActiveTool = 'mask' | 'none';
type MobileView = 'input' | 'output';

const App: React.FC = () => {
  const { t } = useTranslation();
  const [transformations, setTransformations] = useState<Transformation[]>(() => {
    try {
      const savedOrder = localStorage.getItem('transformationOrder');
      if (savedOrder) {
        const orderedKeys = JSON.parse(savedOrder) as string[];
        const transformationMap = new Map(TRANSFORMATIONS.map(t => [t.key, t]));
        
        const orderedTransformations = orderedKeys
          .map(key => transformationMap.get(key))
          .filter((t): t is Transformation => !!t);

        const savedKeysSet = new Set(orderedKeys);
        const newTransformations = TRANSFORMATIONS.filter(t => !savedKeysSet.has(t.key));
        
        return [...orderedTransformations, ...newTransformations];
      }
    } catch (e) {
      console.error("Failed to load or parse transformation order from localStorage", e);
    }
    return TRANSFORMATIONS;
  });

  const [selectedTransformation, setSelectedTransformation] = useState<Transformation | null>(null);
  const [primaryImageUrl, setPrimaryImageUrl] = useState<string | null>(null);
  const [primaryFile, setPrimaryFile] = useState<File | null>(null);
  const [secondaryImageUrl, setSecondaryImageUrl] = useState<string | null>(null);
  const [secondaryFile, setSecondaryFile] = useState<File | null>(null);
  const [multiImageUrls, setMultiImageUrls] = useState<string[]>([]);
  const [generatedContent, setGeneratedContent] = useState<GeneratedContent | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [loadingMessage, setLoadingMessage] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [maskDataUrl, setMaskDataUrl] = useState<string | null>(null);
  const [previewImageUrl, setPreviewImageUrl] = useState<string | null>(null);
  const [customPrompt, setCustomPrompt] = useState<string>('');
  const [aspectRatio, setAspectRatio] = useState<'16:9' | '9:16'>('16:9');
  const [imageAspectRatio, setImageAspectRatio] = useState<'1:1' | '16:9' | '9:16' | '4:3' | '3:4'>('1:1');
  const [activeTool, setActiveTool] = useState<ActiveTool>('none');
  const [history, setHistory] = useState<GeneratedContent[]>([]);
  const [isHistoryPanelOpen, setIsHistoryPanelOpen] = useState<boolean>(false);
  const [activeCategory, setActiveCategory] = useState<Transformation | null>(null);
  const [imageOptions, setImageOptions] = useState<string[] | null>(null);
  const [selectedOption, setSelectedOption] = useState<string | null>(null);
  const [mobileView, setMobileView] = useState<MobileView>('input');
  
  useEffect(() => {
    try {
      const orderToSave = transformations.map(t => t.key);
      localStorage.setItem('transformationOrder', JSON.stringify(orderToSave));
    } catch (e) {
      console.error("Failed to save transformation order to localStorage", e);
    }
  }, [transformations]);
  
  // Cleanup blob URLs on unmount or when dependencies change
  useEffect(() => {
    return () => {
        history.forEach(item => {
            if (item.videoUrl) {
                URL.revokeObjectURL(item.videoUrl);
            }
        });
        if (generatedContent?.videoUrl) {
            URL.revokeObjectURL(generatedContent.videoUrl);
        }
    };
  }, [history, generatedContent]);


  const handleSelectTransformation = (transformation: Transformation) => {
    setSelectedTransformation(transformation);
    setGeneratedContent(null);
    setError(null);
    setImageOptions(null);
    setSelectedOption(null);
    setMobileView('input');
    if (transformation.prompt !== 'CUSTOM') {
      setCustomPrompt('');
    }
    // If switching to a multi-image effect and a primary image exists from a previous step,
    // automatically use it as the first image in the uploader.
    if (transformation.maxImages && primaryImageUrl && multiImageUrls.length === 0) {
        setMultiImageUrls([primaryImageUrl]);
    }
  };

  const handlePrimaryImageSelect = useCallback((file: File, dataUrl: string) => {
    setPrimaryFile(file);
    setPrimaryImageUrl(dataUrl);
    setGeneratedContent(null);
    setError(null);
    setMaskDataUrl(null);
    setActiveTool('none');
    setImageOptions(null);
    setSelectedOption(null);
  }, []);

  const handleSecondaryImageSelect = useCallback((file: File, dataUrl: string) => {
    setSecondaryFile(file);
    setSecondaryImageUrl(dataUrl);
    setGeneratedContent(null);
    setError(null);
  }, []);
  
  const handleClearPrimaryImage = () => {
    setPrimaryImageUrl(null);
    setPrimaryFile(null);
    setMultiImageUrls([]);
    setGeneratedContent(null);
    setError(null);
    setMaskDataUrl(null);
    setActiveTool('none');
    setImageOptions(null);
    setSelectedOption(null);
  };
  
  const handleClearSecondaryImage = () => {
    setSecondaryImageUrl(null);
    setSecondaryFile(null);
  };
  
  const handleGenerateDynamicVideo = useCallback(async () => {
    if (!selectedTransformation || !selectedTransformation.videoPrompt || !selectedOption) {
      setError(t('app.error.selectOneToAnimate'));
      return;
    }

    setIsLoading(true);
    setError(null);
    setGeneratedContent(null);
    setImageOptions(null);
    setMobileView('output');

    try {
        const mimeType = selectedOption.split(';')[0].split(':')[1] ?? 'image/png';
        const base64 = selectedOption.split(',')[1];
        const imagePayload = { base64, mimeType };

        const videoDownloadUrl = await generateVideo(
            selectedTransformation.videoPrompt,
            imagePayload,
            '9:16', // Changed from '1:1' as it's not supported for video
            (message) => setLoadingMessage(message)
        );

        setLoadingMessage(t('app.loading.videoFetching'));
        const response = await fetch(videoDownloadUrl);
        if (!response.ok) throw new Error(`Failed to download video file. Status: ${response.statusText}`);
        
        const blob = await response.blob();
        const objectUrl = URL.createObjectURL(blob);

        const result: GeneratedContent = { imageUrl: null, text: null, videoUrl: objectUrl };
        setGeneratedContent(result);
        setHistory(prev => [result, ...prev]);

    } catch (err) {
        console.error(err);
        setError(err instanceof Error ? err.message : t('app.error.unknown'));
    } finally {
        setIsLoading(false);
        setLoadingMessage('');
        setSelectedOption(null);
    }
  }, [selectedTransformation, selectedOption, t]);


  const handleGenerateVideo = useCallback(async () => {
    if (!selectedTransformation) return;

    const promptToUse = customPrompt;
    if (!promptToUse.trim()) {
        setError(t('app.error.enterPrompt'));
        return;
    }

    setIsLoading(true);
    setError(null);
    setGeneratedContent(null);
    setMobileView('output');

    try {
        let imagePayload = null;
        if (primaryImageUrl) {
            const primaryMimeType = primaryImageUrl.split(';')[0].split(':')[1] ?? 'image/png';
            const primaryBase64 = primaryImageUrl.split(',')[1];
            imagePayload = { base64: primaryBase64, mimeType: primaryMimeType };
        }

        const videoDownloadUrl = await generateVideo(
            promptToUse,
            imagePayload,
            aspectRatio,
            (message) => setLoadingMessage(message) // Progress callback
        );

        setLoadingMessage(t('app.loading.videoFetching'));
        const response = await fetch(videoDownloadUrl);
        if (!response.ok) {
            throw new Error(`Failed to download video file. Status: ${response.statusText}`);
        }
        const blob = await response.blob();
        const objectUrl = URL.createObjectURL(blob);

        const result: GeneratedContent = {
            imageUrl: null,
            text: null,
            videoUrl: objectUrl
        };

        setGeneratedContent(result);
        setHistory(prev => [result, ...prev]);

    } catch (err) {
        console.error(err);
        setError(err instanceof Error ? err.message : t('app.error.unknown'));
    } finally {
        setIsLoading(false);
        setLoadingMessage('');
    }
  }, [selectedTransformation, customPrompt, primaryImageUrl, aspectRatio, t]);
  
  const applyWatermarks = useCallback(async (imageUrl: string | null) => {
    if (!imageUrl) return null;
    try {
      const invisiblyWatermarked = await embedWatermark(imageUrl, "Nano Bananary｜ZHO");
      const visiblyWatermarked = await addVisibleWatermark(invisiblyWatermarked, "Nano Bananary｜ZHO");
      return visiblyWatermarked;
    } catch (err) {
      console.error("Failed to apply watermarks", err);
      return imageUrl; // Return original image on any watermarking failure
    }
  }, []);

  const handleGenerateImage = useCallback(async () => {
    const promptToUse = selectedTransformation?.prompt === 'CUSTOM' ? customPrompt : selectedTransformation?.prompt;

    if (!selectedTransformation || !promptToUse?.trim()) {
        setError(t('app.error.uploadAndSelect'));
        return;
    }

    setIsLoading(true);
    setError(null);
    setGeneratedContent(null);
    setLoadingMessage('');
    setMobileView('output');

    try {
        if (selectedTransformation.isMultiStepVideo) {
             if (multiImageUrls.length === 0) {
                setError(t('app.error.uploadOne'));
                setIsLoading(false);
                return;
            }
            setLoadingMessage(t('app.loading.generatingOptions'));
            const imageParts = multiImageUrls.map(url => ({
                base64: url.split(',')[1],
                mimeType: url.split(';')[0].split(':')[1] ?? 'image/png'
            }));
            const results = await generateImageEditsBatch(promptToUse, imageParts);
            const watermarkedResults = await Promise.all(results.map(url => applyWatermarks(url)));
            setImageOptions(watermarkedResults.filter((url): url is string => !!url));
            setSelectedOption(null);
            return;
        }

        // Text-to-image for custom prompt when no images are provided
        if (selectedTransformation.key === 'customPrompt' && !primaryImageUrl) {
            const result = await generateImageFromText(promptToUse, imageAspectRatio);
            result.imageUrl = await applyWatermarks(result.imageUrl);
            setGeneratedContent(result);
            setHistory(prev => [result, ...prev]);
            return; 
        }

        const maskBase64 = maskDataUrl ? maskDataUrl.split(',')[1] : null;

        if (selectedTransformation.maxImages) {
            if (multiImageUrls.length === 0) {
                setError(t('app.error.uploadOne'));
                setIsLoading(false);
                return;
            }
            const imageParts = multiImageUrls.map(url => ({
                base64: url.split(',')[1],
                mimeType: url.split(';')[0].split(':')[1] ?? 'image/png'
            }));
            const result = await editImage(promptToUse, imageParts, null);
            result.imageUrl = await applyWatermarks(result.imageUrl);
            setGeneratedContent(result);
            setHistory(prev => [result, ...prev]);

        } else if (selectedTransformation.isTwoStep) {
            if (!primaryImageUrl || !secondaryImageUrl) {
                setError(t('app.error.uploadBoth'));
                setIsLoading(false);
                return;
            }
            setLoadingMessage(t('app.loading.step1'));
            const primaryPart = [{ base64: primaryImageUrl.split(',')[1], mimeType: primaryImageUrl.split(';')[0].split(':')[1] ?? 'image/png' }];
            const stepOneResult = await editImage(promptToUse, primaryPart, null);

            if (!stepOneResult.imageUrl) throw new Error("Step 1 (line art) failed to generate an image.");
            
            stepOneResult.imageUrl = await applyWatermarks(stepOneResult.imageUrl);

            setLoadingMessage(t('app.loading.step2'));
            const stepOneImageBase64 = stepOneResult.imageUrl.split(',')[1];
            const stepOneImageMimeType = stepOneResult.imageUrl.split(';')[0].split(':')[1] ?? 'image/png';
            
            const primaryImage = await loadImage(primaryImageUrl);
            const resizedSecondaryImageUrl = await resizeImageToMatch(secondaryImageUrl, primaryImage);
            
            const stepTwoParts = [
                { base64: stepOneImageBase64, mimeType: stepOneImageMimeType },
                { base64: resizedSecondaryImageUrl.split(',')[1], mimeType: resizedSecondaryImageUrl.split(';')[0].split(':')[1] ?? 'image/png' }
            ];
            
            const stepTwoResult = await editImage(selectedTransformation.stepTwoPrompt!, stepTwoParts, null);
            
            stepTwoResult.imageUrl = await applyWatermarks(stepTwoResult.imageUrl);

            const finalResult = { ...stepTwoResult, secondaryImageUrl: stepOneResult.imageUrl };
            setGeneratedContent(finalResult);
            setHistory(prev => [finalResult, ...prev]);

        } else {
             if (!primaryImageUrl) {
                setError(t('app.error.uploadAndSelect'));
                setIsLoading(false);
                return;
             }
             if (selectedTransformation.isMultiImage && !selectedTransformation.isSecondaryOptional && !secondaryImageUrl) {
                setError(t('app.error.uploadBoth'));
                setIsLoading(false);
                return;
             }

            let imageParts = [{ 
                base64: primaryImageUrl.split(',')[1], 
                mimeType: primaryImageUrl.split(';')[0].split(':')[1] ?? 'image/png'
            }];
            if (selectedTransformation.isMultiImage && secondaryImageUrl) {
                imageParts.push({
                    base64: secondaryImageUrl.split(',')[1],
                    mimeType: secondaryImageUrl.split(';')[0].split(':')[1] ?? 'image/png'
                });
            }

            setLoadingMessage(t('app.loading.default'));
            const result = await editImage(promptToUse, imageParts, maskBase64);

            result.imageUrl = await applyWatermarks(result.imageUrl);

            setGeneratedContent(result);
            setHistory(prev => [result, ...prev]);
        }
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : t('app.error.unknown'));
    } finally {
      setIsLoading(false);
      setLoadingMessage('');
    }
  }, [primaryImageUrl, secondaryImageUrl, selectedTransformation, maskDataUrl, customPrompt, multiImageUrls, t, imageAspectRatio, applyWatermarks]);
  
  const handleGenerate = useCallback(() => {
    if (selectedTransformation?.isVideo) {
      handleGenerateVideo();
    } else {
      handleGenerateImage();
    }
  }, [selectedTransformation, handleGenerateVideo, handleGenerateImage]);


  const handleUseImageAsInput = useCallback(async (imageUrl: string) => {
    if (!imageUrl) return;

    try {
      const newFile = await dataUrlToFile(imageUrl, `edited-${Date.now()}.png`);
      setPrimaryFile(newFile);
      setPrimaryImageUrl(imageUrl);
      setGeneratedContent(null);
      setError(null);
      setMaskDataUrl(null);
      setActiveTool('none');
      setSecondaryFile(null);
      setSecondaryImageUrl(null);
      setMultiImageUrls([imageUrl]);
      setSelectedTransformation(null); 
      setActiveCategory(null);
      setImageOptions(null);
      setSelectedOption(null);
      setMobileView('input');
    } catch (err) {
      console.error("Failed to use image as input:", err);
      setError(t('app.error.useAsInputFailed'));
    }
  }, [t]);
  
  const toggleHistoryPanel = () => setIsHistoryPanelOpen(prev => !prev);
  
  const handleUseHistoryImageAsInput = (imageUrl: string) => {
      handleUseImageAsInput(imageUrl);
      setIsHistoryPanelOpen(false);
  };
  
  const handleDownloadFromHistory = (url: string, type: string) => {
      const fileExtension = type.includes('video') ? 'mp4' : (url.split(';')[0].split('/')[1] || 'png');
      const filename = `${type}-${Date.now()}.${fileExtension}`;
      downloadImage(url, filename);
  };

  const handleBackToSelection = () => {
    setSelectedTransformation(null);
    setImageOptions(null);
    setSelectedOption(null);
    setMobileView('input');
  };

  const handleResetApp = () => {
    setSelectedTransformation(null);
    setPrimaryImageUrl(null);
    setPrimaryFile(null);
    setSecondaryImageUrl(null);
    setSecondaryFile(null);
    setMultiImageUrls([]);
    setGeneratedContent(null);
    setError(null);
    setIsLoading(false);
    setMaskDataUrl(null);
    setCustomPrompt('');
    setActiveTool('none');
    setActiveCategory(null);
    setImageOptions(null);
    setSelectedOption(null);
    setMobileView('input');
  };

  const handleOpenPreview = (url: string) => setPreviewImageUrl(url);
  const handleClosePreview = () => setPreviewImageUrl(null);
  
  const toggleMaskTool = () => {
    setActiveTool(current => (current === 'mask' ? 'none' : 'mask'));
  };
  
  const isCustomPromptEmpty = selectedTransformation?.prompt === 'CUSTOM' && !customPrompt.trim();
  
  let isGenerateDisabled = true;
    if (selectedTransformation) {
        if (selectedTransformation.key === 'customPrompt') {
            isGenerateDisabled = isLoading || isCustomPromptEmpty;
        } else if (selectedTransformation.isVideo) {
            isGenerateDisabled = isLoading || !customPrompt.trim();
        } else if (selectedTransformation.maxImages) {
            isGenerateDisabled = isLoading || multiImageUrls.length === 0;
        } else {
            let imagesReady = false;
            if (selectedTransformation.isMultiImage) {
                if (selectedTransformation.isSecondaryOptional) {
                    imagesReady = !!primaryImageUrl;
                } else {
                    imagesReady = !!primaryImageUrl && !!secondaryImageUrl;
                }
            } else {
                imagesReady = !!primaryImageUrl;
            }
            isGenerateDisabled = isLoading || isCustomPromptEmpty || !imagesReady;
        }
    }

  const renderInputUI = () => {
    if (!selectedTransformation) return null;

    if (selectedTransformation.maxImages) {
        return (
            <MultiImageGridUploader
                imageUrls={multiImageUrls}
                onImagesChange={setMultiImageUrls}
                maxImages={selectedTransformation.maxImages}
            />
        );
    }

    if (selectedTransformation.isVideo) {
      return (
        <>
          <textarea
            value={customPrompt}
            onChange={(e) => setCustomPrompt(e.target.value)}
            placeholder={t('transformations.video.promptPlaceholder')}
            rows={4}
            className="w-full mt-2 p-3 bg-[var(--bg-secondary)] border border-[var(--border-primary)] rounded-lg focus:ring-2 focus:ring-[var(--accent-primary)] focus:border-[var(--accent-primary)] transition-colors placeholder-[var(--text-tertiary)]"
          />
          <div className="mt-4">
            <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-2">{t('transformations.video.aspectRatio')}</h3>
            <div className="grid grid-cols-2 gap-2">
              {(['16:9', '9:16'] as const).map(ratio => (
                <button
                  key={ratio}
                  onClick={() => setAspectRatio(ratio)}
                  className={`py-2 px-3 text-sm font-semibold rounded-md transition-colors duration-200 ${
                    aspectRatio === ratio ? 'bg-gradient-to-r from-[var(--accent-primary)] to-[var(--accent-secondary)] text-[var(--text-on-accent)]' : 'bg-[rgba(107,114,128,0.2)] hover:bg-[rgba(107,114,128,0.4)]'
                  }`}
                >
                  {t(ratio === '16:9' ? 'transformations.video.landscape' : 'transformations.video.portrait')}
                </button>
              ))}
            </div>
          </div>
          <div className="mt-4">
             <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-2">{t('transformations.effects.customPrompt.uploader2Title')}</h3>
            <ImageEditorCanvas
                onImageSelect={handlePrimaryImageSelect}
                initialImageUrl={primaryImageUrl}
                onMaskChange={() => {}}
                onClearImage={handleClearPrimaryImage}
                isMaskToolActive={false}
            />
          </div>
        </>
      );
    }

    if (selectedTransformation.isMultiImage) {
      return (
        <MultiImageUploader
          onPrimarySelect={handlePrimaryImageSelect}
          onSecondarySelect={handleSecondaryImageSelect}
          primaryImageUrl={primaryImageUrl}
          secondaryImageUrl={secondaryImageUrl}
          onClearPrimary={handleClearPrimaryImage}
          onClearSecondary={handleClearSecondaryImage}
          primaryTitle={selectedTransformation.primaryUploaderTitle ? t(selectedTransformation.primaryUploaderTitle) : undefined}
          primaryDescription={selectedTransformation.primaryUploaderDescription ? t(selectedTransformation.primaryUploaderDescription) : undefined}
          secondaryTitle={selectedTransformation.secondaryUploaderTitle ? t(selectedTransformation.secondaryUploaderTitle) : undefined}
          secondaryDescription={selectedTransformation.secondaryUploaderDescription ? t(selectedTransformation.secondaryUploaderDescription) : undefined}
        />
      );
    }

    return (
      <>
        <ImageEditorCanvas
          onImageSelect={handlePrimaryImageSelect}
          initialImageUrl={primaryImageUrl}
          onMaskChange={setMaskDataUrl}
          onClearImage={handleClearPrimaryImage}
          isMaskToolActive={activeTool === 'mask'}
        />
        {primaryImageUrl && (
          <div className="mt-4">
            <button
              onClick={toggleMaskTool}
              className={`w-full flex items-center justify-center gap-2 py-2 px-3 text-sm font-semibold rounded-md transition-colors duration-200 ${
                activeTool === 'mask' ? 'bg-gradient-to-r from-[var(--accent-primary)] to-[var(--accent-secondary)] text-[var(--text-on-accent)]' : 'bg-[rgba(107,114,128,0.2)] hover:bg-[rgba(107,114,128,0.4)]'
              }`}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.5L15.232 5.232z" /></svg>
              <span>{t('imageEditor.drawMask')}</span>
            </button>
          </div>
        )}
      </>
    );
  };


  return (
    <div className="min-h-screen bg-[var(--bg-primary)] text-[var(--text-primary)] font-sans">
      <header className="bg-[var(--bg-card-alpha)] backdrop-blur-lg sticky top-0 z-20 p-4 border-b border-[var(--border-primary)]">
        <div className="container mx-auto flex justify-between items-center">
          <h1 
            className="text-xl sm:text-2xl font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-[var(--accent-primary)] to-[var(--accent-secondary)] cursor-pointer" 
            onClick={handleResetApp}
          >
            {t('app.title')}
          </h1>
          <div className="flex items-center gap-2 md:gap-4">
            <button
              onClick={toggleHistoryPanel}
              className="flex items-center justify-center w-10 h-10 text-sm font-semibold text-[var(--text-primary)] bg-[rgba(107,114,128,0.2)] rounded-full hover:bg-[rgba(107,114,128,0.4)] transition-colors duration-200"
              aria-label={t('app.history')}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
              </svg>
            </button>
            <LanguageSwitcher />
            <ThemeSwitcher />
          </div>
        </div>
      </header>

      <main>
        {!selectedTransformation ? (
          <TransformationSelector 
            transformations={transformations} 
            onSelect={handleSelectTransformation} 
            hasPreviousResult={!!primaryImageUrl || multiImageUrls.length > 0}
            onOrderChange={setTransformations}
            activeCategory={activeCategory}
            setActiveCategory={setActiveCategory}
          />
        ) : (
          <div className="container mx-auto p-4 md:p-8 animate-fade-in">
            <div className="mb-4 md:mb-8">
              <button
                onClick={handleBackToSelection}
                className="flex items-center gap-2 text-[var(--accent-primary)] hover:text-[var(--accent-primary-hover)] transition-colors duration-200 py-2 px-4 rounded-lg hover:bg-[rgba(107,114,128,0.1)]"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                {t('app.chooseAnotherEffect')}
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-8">
              {/* Input Column */}
              <div className={`flex-col gap-6 p-4 sm:p-6 bg-[var(--bg-card-alpha)] backdrop-blur-lg rounded-xl border border-[var(--border-primary)] shadow-2xl shadow-black/20 ${mobileView === 'input' ? 'flex' : 'hidden md:flex'}`}>
                <div>
                  <div className="mb-4">
                    <h2 className="text-lg md:text-xl font-semibold mb-1 text-[var(--accent-primary)] flex items-center gap-3">
                      <span className="text-3xl">{selectedTransformation.emoji}</span>
                      {t(selectedTransformation.titleKey)}
                    </h2>
                    {selectedTransformation.prompt !== 'CUSTOM' ? (
                       <p className="text-[var(--text-secondary)]">{t(selectedTransformation.descriptionKey)}</p>
                    ) : (
                      !selectedTransformation.isVideo && <p className="text-[var(--text-secondary)]">{t(selectedTransformation.descriptionKey)}</p>
                    )}
                  </div>
                  
                  {selectedTransformation.prompt === 'CUSTOM' && !selectedTransformation.isVideo && (
                     <>
                        <textarea
                            value={customPrompt}
                            onChange={(e) => setCustomPrompt(e.target.value)}
                            placeholder="e.g., A robot holding a red skateboard."
                            rows={3}
                            className="w-full -mt-2 mb-4 p-3 bg-[var(--bg-secondary)] border border-[var(--border-primary)] rounded-lg focus:ring-2 focus:ring-[var(--accent-primary)] focus:border-[var(--accent-primary)] transition-colors placeholder-[var(--text-tertiary)]"
                        />
                         {!primaryImageUrl && (
                            <div className="mb-4 animate-fade-in-fast">
                                <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-2">{t('app.aspectRatio')}</h3>
                                <div className="grid grid-cols-3 sm:grid-cols-5 gap-2">
                                    {(['1:1', '16:9', '9:16', '4:3', '3:4'] as const).map(ratio => (
                                        <button
                                          key={ratio}
                                          onClick={() => setImageAspectRatio(ratio)}
                                          className={`py-2 px-3 text-xs font-semibold rounded-md transition-colors duration-200 ${
                                            imageAspectRatio === ratio ? 'bg-gradient-to-r from-[var(--accent-primary)] to-[var(--accent-secondary)] text-[var(--text-on-accent)]' : 'bg-[rgba(107,114,128,0.2)] hover:bg-[rgba(107,114,128,0.4)]'
                                          }`}
                                        >
                                          {ratio}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        )}
                    </>
                  )}
                  
                  {renderInputUI()}
                  
                   <button
                    onClick={handleGenerate}
                    disabled={isGenerateDisabled}
                    className="w-full mt-6 py-3 px-4 bg-gradient-to-r from-[var(--accent-primary)] to-[var(--accent-secondary)] text-[var(--text-on-accent)] font-semibold rounded-lg shadow-lg shadow-[var(--accent-shadow)] hover:from-[var(--accent-primary-hover)] hover:to-[var(--accent-secondary-hover)] disabled:bg-[var(--bg-disabled)] disabled:from-[var(--bg-disabled)] disabled:to-[var(--bg-disabled)] disabled:text-[var(--text-disabled)] disabled:shadow-none disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center gap-2"
                  >
                    {isLoading ? (
                      <>
                        <svg className="animate-spin -ml-1 mr-2 h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        <span>{t('app.generating')}</span>
                      </>
                    ) : (
                      <>
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                          <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                        </svg>
                        <span>{t('app.generateImage')}</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Output Column */}
              <div className={`flex-col p-4 sm:p-6 bg-[var(--bg-card-alpha)] backdrop-blur-lg rounded-xl border border-[var(--border-primary)] shadow-2xl shadow-black/20 ${mobileView === 'output' ? 'flex' : 'hidden md:flex'}`}>
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-lg md:text-xl font-semibold text-[var(--accent-primary)]">{t('app.result')}</h2>
                  <button
                      onClick={() => setMobileView('input')}
                      className="md:hidden flex items-center gap-1.5 text-sm py-1 px-2.5 rounded-md text-[var(--accent-primary)] hover:bg-[rgba(107,114,128,0.1)] transition-colors"
                      aria-label={t('app.edit')}
                  >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor"><path d="M17.414 2.586a2 2 0 00-2.828 0L7 10.172V13h2.828l7.586-7.586a2 2 0 000-2.828z" /><path fillRule="evenodd" d="M2 6a2 2 0 012-2h4a1 1 0 010 2H4v10h10v-4a1 1 0 112 0v4a2 2 0 01-2-2H4a2 2 0 01-2-2V6z" clipRule="evenodd" /></svg>
                      <span>{t('app.edit')}</span>
                  </button>
                </div>
                {isLoading && <div className="flex-grow flex items-center justify-center"><LoadingSpinner message={loadingMessage} /></div>}
                {error && <div className="flex-grow flex items-center justify-center w-full"><ErrorMessage message={error} /></div>}
                
                {!isLoading && !error && imageOptions && (
                    <div className="flex-grow flex flex-col items-center justify-center gap-4 animate-fade-in">
                        <h3 className="text-md font-semibold text-[var(--text-secondary)]">{t('app.chooseYourShot')}</h3>
                        <div className="grid grid-cols-2 gap-2 w-full">
                            {imageOptions.map((url, index) => (
                                <button key={index} onClick={() => setSelectedOption(url)} className={`relative rounded-lg overflow-hidden border-2 transition-all duration-200 ${selectedOption === url ? 'border-[var(--accent-primary)] scale-105 shadow-lg' : 'border-transparent hover:scale-105'}`}>
                                    <img src={url} alt={`Option ${index + 1}`} className="aspect-square object-contain bg-[var(--bg-secondary)]" />
                                    {selectedOption === url && (
                                        <div className="absolute inset-0 bg-[var(--accent-primary)]/30 flex items-center justify-center">
                                            <svg className="h-8 w-8 text-white" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                                                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                            </svg>
                                        </div>
                                    )}
                                </button>
                            ))}
                        </div>
                        <div className="w-full grid grid-cols-2 gap-4 mt-4">
                            <button onClick={handleGenerateImage} className="w-full py-2 px-4 bg-[rgba(107,114,128,0.2)] hover:bg-[rgba(107,114,128,0.4)] text-[var(--text-primary)] font-semibold rounded-lg transition-colors flex items-center justify-center gap-2">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h5M20 20v-5h-5" />
                                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 10a8 8 0 0114.95-2.95l-2.45 1.76" />
                                  <path strokeLinecap="round" strokeLinejoin="round" d="M20 14a8 8 0 01-14.95 2.95l2.45-1.76" />
                                </svg>
                                {t('app.regenerate')}
                            </button>
                             <button onClick={handleGenerateDynamicVideo} disabled={!selectedOption || isLoading} className="w-full py-2 px-4 bg-gradient-to-r from-[var(--accent-primary)] to-[var(--accent-secondary)] text-[var(--text-on-accent)] font-semibold rounded-lg shadow-lg shadow-[var(--accent-shadow)] hover:from-[var(--accent-primary-hover)] hover:to-[var(--accent-secondary-hover)] disabled:bg-[var(--bg-disabled)] disabled:from-[var(--bg-disabled)] disabled:to-[var(--bg-disabled)] disabled:text-[var(--text-disabled)] disabled:shadow-none disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center gap-2">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path d="M2 6a2 2 0 012-2h6a2 2 0 012 2v8a2 2 0 01-2 2H4a2 2 0 01-2-2V6zM14.553 7.106A1 1 0 0014 8v4a1 1 0 001.553.832l3-2a1 1 0 000-1.664l-3-2z" /></svg>
                                {t('app.createVideo')}
                            </button>
                        </div>
                    </div>
                )}

                {!isLoading && !error && !imageOptions && generatedContent && (
                    <ResultDisplay 
                        content={generatedContent} 
                        onUseImageAsInput={handleUseImageAsInput}
                        onImageClick={handleOpenPreview}
                        originalImageUrl={primaryImageUrl}
                        onRegenerate={handleGenerate}
                        isGenerating={isLoading}
                    />
                )}
                {!isLoading && !error && !imageOptions && !generatedContent && (
                  <div className="flex-grow flex flex-col items-center justify-center text-center text-[var(--text-tertiary)]">
                    <svg xmlns="http://www.w3.org/2000/svg" className="mx-auto h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <p className="mt-2">{t('app.yourImageWillAppear')}</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>
      <ImagePreviewModal imageUrl={previewImageUrl} onClose={handleClosePreview} />
      <HistoryPanel
        isOpen={isHistoryPanelOpen}
        onClose={toggleHistoryPanel}
        history={history}
        onUseImage={handleUseHistoryImageAsInput}
        onDownload={handleDownloadFromHistory}
      />
    </div>
  );
};

// Add fade-in animation for view transitions
const style = document.createElement('style');
style.innerHTML = `
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .animate-fade-in {
    animation: fadeIn 0.4s ease-out forwards;
  }
  @keyframes fadeInFast {
    from { opacity: 0; }
    to { opacity: 1; }
  }
  .animate-fade-in-fast {
    animation: fadeInFast 0.2s ease-out forwards;
  }
`;
document.head.appendChild(style);


export default App;