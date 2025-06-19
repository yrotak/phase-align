#include "imgui.h"
#include "imgui_impl_sdl.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <SDL.h>
#include <cmath>
#include <thread>
#include <complex>
#include <algorithm>
#include <mutex>
#include <chrono>
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <SDL_opengles2.h>
#else
#include <SDL_opengl.h>
#endif

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include "processing.hpp"
#include "imfilebrowser.h"

struct AudioPlayback
{
    float *samples = nullptr;
    uint64_t totalSampleCount = 0;
    uint64_t currentIndex = 0;
    unsigned int channels = 1;
};

std::vector<AudioPlayback> g_playbacks;
SDL_AudioDeviceID g_audioDevice = 0;

static uint64_t g_elapsedSamples = 0;
static uint64_t g_maxSamples = 0;
static bool g_isPlaying = false;
int g_refAudioIndex = 0;
float g_findStartThreshold = 50.f;
float g_findStartSize = 100.f;
int g_findStartFrame = -1;

std::string g_errorPopup = "Created by https://github.com/yrotak";
int g_errorPopupTimer = -100;
std::vector<int> g_framesStartSync;

bool g_findStartProcessing = false;
bool g_syncStartProcessing = false;
bool g_cutExcessProcessing = false;
bool g_dtwProcessing = false;
bool g_correlationProcessing = false;

double g_findStartTime = 0.0;
double g_syncStartTime = 0.0;
double g_cutExcessTime = 0.0;
double g_dtwTime = 0.0;
double g_correlationTime = 0.0;

struct CorrelationResult
{
    int shift;
    float maxCorrelation;
};
std::vector<CorrelationResult> g_correlationResults;
std::mutex g_audioMutex;

void AudioCallback(void *userdata, Uint8 *stream, int len)
{
    float *out = (float *)stream;
    int floatCount = len / sizeof(float);

    for (int i = 0; i < floatCount; ++i)
    {
        float mixedSample = 0.0f;
        int activeTracks = 0;

        for (auto &playback : g_playbacks)
        {
            if (playback.currentIndex < playback.totalSampleCount)
            {
                mixedSample += playback.samples[playback.currentIndex++];
                activeTracks++;

                g_elapsedSamples = playback.currentIndex;
            }
        }

        out[i] = activeTracks > 0 ? mixedSample / activeTracks : 0.0f;
    }
}

void StartAudioSystemIfNeeded(int sampleRate, int channels)
{
    if (g_audioDevice != 0)
        return;

    SDL_AudioSpec want = {};
    want.freq = sampleRate;
    want.format = AUDIO_F32SYS;
    want.channels = channels;
    want.samples = 4096;
    want.callback = AudioCallback;

    g_audioDevice = SDL_OpenAudioDevice(NULL, 0, &want, NULL, 0);
    if (g_audioDevice == 0)
    {
        std::cerr << "Failed to open audio: " << SDL_GetError() << std::endl;
        return;
    }

    SDL_PauseAudioDevice(g_audioDevice, 0);
}

void AddAudioToPlayback(const AudioFile &audio)
{
    AudioPlayback pb;
    pb.samples = audio.samples;
    pb.totalSampleCount = audio.totalSampleCount;
    pb.currentIndex = 0;
    pb.channels = audio.channels;
    g_playbacks.push_back(pb);

    if (audio.totalSampleCount > g_maxSamples)
    {
        g_maxSamples = audio.totalSampleCount;
    }
}

const float g_waveformGain = 200.0f;

void DrawAudioOnTimeline(const AudioFile &audioFile, int width, int height, drwav_uint64 maxFrames, ImU32 color = IM_COL32(0, 255, 0, 255))
{
    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    ImVec2 origin = ImGui::GetCursorScreenPos();

    float framesPerPixel = static_cast<float>(maxFrames) / width;
    float centerY = origin.y + height / 2.0f;

    for (int x = 0; x < width; ++x)
    {
        drwav_uint64 startFrame = static_cast<drwav_uint64>(x * framesPerPixel);
        drwav_uint64 endFrame = static_cast<drwav_uint64>((x + 1) * framesPerPixel);

        if (startFrame >= audioFile.totalSampleCount / audioFile.channels)
        {
            break;
        }

        float minSample = 0.0f;
        float maxSample = 0.0f;
        bool hasData = false;

        for (drwav_uint64 frame = startFrame; frame < endFrame; ++frame)
        {
            if (frame >= audioFile.totalSampleCount / audioFile.channels)
                break;

            float sample = audioFile.samples[frame * audioFile.channels];

            if (!hasData)
            {
                minSample = sample;
                maxSample = sample;
                hasData = true;
            }
            else
            {
                if (sample < minSample)
                    minSample = sample;
                if (sample > maxSample)
                    maxSample = sample;
            }
        }

        float yMin, yMax;

        if (hasData)
        {
            yMin = centerY - minSample * g_waveformGain;
            yMax = centerY - maxSample * g_waveformGain;
        }
        else
        {
            yMin = centerY;
            yMax = centerY;
        }

        if (yMin == yMax)
        {
            yMin -= 0.5f;
            yMax += 0.5f;
        }

        draw_list->AddLine(
            ImVec2(origin.x + x, yMin),
            ImVec2(origin.x + x, yMax),
            color, 2.f);
    }
}

void DrawTimelineScale(float width, float height, int sampleRate, drwav_uint64 maxFrames)
{
    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    ImVec2 origin = ImGui::GetCursorScreenPos();
    float seconds = static_cast<float>(maxFrames) / sampleRate;

    for (float s = 0; s < seconds; s += 0.5f)
    {
        float x = (s / seconds) * width;
        draw_list->AddLine(ImVec2(origin.x + x, origin.y),
                           ImVec2(origin.x + x, origin.y + height),
                           IM_COL32(255, 255, 255, 80));
        draw_list->AddText(ImVec2(origin.x + x + 2, origin.y), IM_COL32_WHITE, (std::to_string(s) + "s").c_str());
    }
    ImGui::Dummy(ImVec2(width, height));
}

void DrawProgressBar(float width, float height, drwav_uint64 maxFrames, int sampleRate, std::vector<AudioFile> &alignedFiles)
{
    if (!g_isPlaying || g_maxSamples == 0)
        return;

    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    ImVec2 origin = ImGui::GetCursorScreenPos();
    float progress = static_cast<float>(g_elapsedSamples) / g_maxSamples;
    if (progress > 1.0f)
        progress = 1.0f;

    float x = progress * width;

    draw_list->AddLine(
        ImVec2(origin.x + x, origin.y),
        ImVec2(origin.x + x, origin.y + height),
        IM_COL32(255, 0, 0, 255),
        2.0f);

    float currentTime = static_cast<float>(g_elapsedSamples) / (sampleRate * alignedFiles[0].channels);
    std::string timeText = std::to_string(currentTime) + "s";
    draw_list->AddText(ImVec2(origin.x + x + 5, origin.y + 5), IM_COL32(255, 255, 0, 255), timeText.c_str());
}
void DrawInitialStart(float width, float height, drwav_uint64 maxFrames, int sampleRate, int startFrame, int sizeFrame = 1)
{

    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    ImVec2 origin = ImGui::GetCursorScreenPos();
    float progress1 = static_cast<float>(startFrame - sizeFrame) / maxFrames;
    if (progress1 > 1.0f)
        progress1 = 1.0f;

    float x1 = progress1 * width;

    float progress2 = static_cast<float>(startFrame + sizeFrame) / maxFrames;
    if (progress2 > 1.0f)
        progress2 = 1.0f;

    float x2 = progress2 * width;

    draw_list->AddRect(
        ImVec2(origin.x + x1, origin.y),
        ImVec2(origin.x + x2, origin.y + height),
        IM_COL32(255, 255, 255, 100),
        0.0f,
        ImDrawFlags_RoundCornersAll,
        2.0f);
}

void DrawCorrelationMarkers(float width, float height, drwav_uint64 maxFrames, const std::vector<CorrelationResult> &results)
{
    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    ImVec2 origin = ImGui::GetCursorScreenPos();

    for (const auto &result : results)
    {
        if (result.shift >= 0)
        {
            float progress = static_cast<float>(result.shift) / maxFrames;
            if (progress > 1.0f)
                progress = 1.0f;

            float x = progress * width;

            ImU32 color = IM_COL32(255, 0, 255, 255);
            if (result.maxCorrelation > 0.9f)
                color = IM_COL32(0, 255, 0, 255);
            else if (result.maxCorrelation > 0.7f)
                color = IM_COL32(0, 200, 100, 255);
            else if (result.maxCorrelation > 0.5f)
                color = IM_COL32(255, 255, 0, 255);
            else
                color = IM_COL32(255, 0, 0, 255);

            draw_list->AddLine(
                ImVec2(origin.x + x, origin.y),
                ImVec2(origin.x + x, origin.y + height),
                color,
                10.0f);

            std::string markerText = std::to_string(result.shift) + " (" +
                                     std::to_string(result.maxCorrelation).substr(0, 4) + ")";
            draw_list->AddText(ImVec2(origin.x + x + 10, origin.y + 15),
                               IM_COL32(255, 255, 255, 255),
                               markerText.c_str());
        }
    }
}

AudioFile readAudioFile(std::string filename, std::string name)
{
    AudioFile audioFile;
    audioFile.samples = drwav_open_file_and_read_pcm_frames_f32(
        filename.c_str(),
        &audioFile.channels,
        &audioFile.sampleRate,
        &audioFile.totalSampleCount,
        NULL);

    audioFile.totalSampleCount = audioFile.totalSampleCount * audioFile.channels;

    if (audioFile.samples == NULL)
    {
        std::cerr << "Failed to read audio file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    audioFile.name = name;
    audioFile.filename = filename;
    return audioFile;
}

drwav_uint64 GetLongestDurationFrames(const std::vector<AudioFile> &files)
{
    drwav_uint64 maxFrames = 0;
    for (const auto &f : files)
    {
        drwav_uint64 frames = f.totalSampleCount / f.channels;
        if (frames > maxFrames)
            maxFrames = frames;
    }
    return maxFrames;
}

void displayError(std::string error)
{
    g_errorPopupTimer = 0;
    g_errorPopup = error;
}
std::string addSyncSuffix(const std::string &path)
{
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".wav")
    {
        return path.substr(0, path.size() - 4) + "_sync.wav";
    }
    else
    {
        return path + "_sync.wav";
    }
}
void saveWavFile(AudioFile &audioFile)
{
    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels = audioFile.channels;
    format.sampleRate = audioFile.sampleRate;
    format.bitsPerSample = 32;

    std::string filename = addSyncSuffix(audioFile.filename);
    drwav wav;
    if (drwav_init_file_write(&wav, filename.c_str(), &format, NULL))
    {
        drwav_uint64 framesWritten = drwav_write_pcm_frames(&wav, audioFile.totalSampleCount, audioFile.samples);
        if (framesWritten != audioFile.totalSampleCount)
        {
            printf("Erreur : Seulement %llu sur %llu frames écrites !\n", framesWritten, audioFile.totalSampleCount);
        }
        drwav_uninit(&wav);
    }
    else
    {
        printf("Erreur : Impossible de créer le fichier %s\n", filename.c_str());
    }
}
int main(int, char **)
{
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER | SDL_INIT_AUDIO) != 0)
    {
        printf("Error: %s\n", SDL_GetError());
        return -1;
    }

#if defined(IMGUI_IMPL_OPENGL_ES2)
    const char *glsl_version = "#version 100";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#elif defined(__APPLE__)
    const char *glsl_version = "#version 150";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
#else
    const char *glsl_version = "#version 130";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#endif
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
    SDL_Window *window = SDL_CreateWindow("Phase Align", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1280, 720, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    SDL_GL_MakeCurrent(window, gl_context);
    SDL_GL_SetSwapInterval(1);
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    ImGui::StyleColorsDark();

    ImGuiStyle *style = &ImGui::GetStyle();

    ImVec4 *colors = style->Colors;
    colors[ImGuiCol_ChildBg] = ImVec4(0.1f, 0.1f, 0.1f, 1.0f);

    ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
    ImGui_ImplOpenGL3_Init(glsl_version);
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    std::vector<AudioFile> audios;

    ImGui::FileBrowser fileDialog;

    fileDialog.SetTitle("Select audio file");
    fileDialog.SetTypeFilters({".wav"});

    drwav_uint64 maxFrames = 0;

    while (true)
    {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT)
                return 0;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x, io.DisplaySize.y));
        ImGui::Begin("##Main", NULL, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar);

        ImGui::BeginChild("#AudiosSettings", ImVec2(io.DisplaySize.x / 3, 200));
        ImGui::Text("Audio Files");
        if (ImGui::Button("Play audios"))
        {
            if (audios.size() == 0)
            {
                displayError("No audio files loaded.");
            }
            else
            {
                g_isPlaying = false;
                g_playbacks.clear();
                g_elapsedSamples = 0;
                g_maxSamples = 0;
                for (size_t i = 0; i < audios.size(); ++i)
                {
                    AddAudioToPlayback(audios[i]);
                }
                StartAudioSystemIfNeeded(audios[0].sampleRate, audios[0].channels);
                g_isPlaying = true;
            }
        }

        if (g_isPlaying)
        {
            bool allFinished = true;
            for (const auto &pb : g_playbacks)
            {
                if (pb.currentIndex < pb.totalSampleCount)
                {
                    allFinished = false;
                    break;
                }
            }

            if (allFinished)
            {
                g_isPlaying = false;
            }
        }

        if (ImGui::Button("Add audio file"))
        {
            fileDialog.Open();
        }

        if (g_correlationProcessing)
        {
            ImGui::Button("Analyzing correlation...");
        }
        else
        {
            if (g_correlationTime > 0.0)
            {
                if (ImGui::Button(("Analyzed in " + std::to_string(g_correlationTime) + " sec (click to reset)").c_str()))
                {
                    g_correlationTime = 0.0f;
                }
            }
            else
            {
                if (ImGui::Button("Check correlation with ref"))
                {
                    if (audios.size() < 2)
                    {
                        displayError("Need at least 2 audio files for correlation analysis");
                    }
                    else
                    {
                        g_correlationProcessing = true;
                        std::thread([&]()
                                    {
                            auto start = std::chrono::high_resolution_clock::now();
                            std::vector<CorrelationResult> results;
                            
                            std::vector<float> ref(
                                audios[g_refAudioIndex].samples,
                                audios[g_refAudioIndex].samples + audios[g_refAudioIndex].totalSampleCount
                            );
                            
                            for (size_t i = 0; i < audios.size(); ++i)
                            {
                                if (i == g_refAudioIndex)
                                {
                                    results.push_back({0, 1.0f});
                                    continue;
                                }
                                
                                std::vector<float> signal(
                                    audios[i].samples,
                                    audios[i].samples + audios[i].totalSampleCount
                                );
                                
                                auto [shift, maxCorrelation] = Processing::findCorrelationShift(ref, signal);
                                results.push_back({shift, maxCorrelation});
                            }
                            
                            auto end = std::chrono::high_resolution_clock::now();
                            std::chrono::duration<double> elapsed = end - start;
                            g_correlationTime = elapsed.count();
                            
                            {
                                std::lock_guard<std::mutex> lock(g_audioMutex);
                                g_correlationResults = results;
                                g_correlationProcessing = false;
                            } })
                            .detach();
                    }
                }
            }
        }

        ImGui::EndChild();
        ImGui::SameLine();
        ImGui::BeginChild("#InitSync", ImVec2(io.DisplaySize.x / 3, 200));
        ImGui::Text("Initial sync");

        ImGui::SliderFloat("Threshold", &g_findStartThreshold, 0.0f, 100.0f, "%1.0f");

        if (g_findStartProcessing)
        {
            ImGui::Button("Processing...");
        }
        else
        {
            if (g_findStartTime > 0.0)
            {
                if (ImGui::Button(("Processed in " + std::to_string(g_findStartTime) + " sec (click to reset)").c_str()))
                {
                    g_findStartTime = 0.0f;
                }
            }
            else
            {
                if (ImGui::Button("Find start of ref audio"))
                {
                    if (g_refAudioIndex < audios.size())
                    {
                        g_findStartProcessing = true;
                        std::thread([&]()
                                    {
                            auto start = std::chrono::high_resolution_clock::now();
                            int startFrame = Processing::findStart(audios[g_refAudioIndex], g_findStartThreshold / 100.f);
                            auto end = std::chrono::high_resolution_clock::now();
                            std::chrono::duration<double> elapsed = end - start;
                            g_findStartTime = elapsed.count();

                            {
                                std::lock_guard<std::mutex> lock(g_audioMutex);
                                if (startFrame >= 0)
                                {
                                    g_findStartFrame = startFrame;
                                }
                                else
                                {
                                    displayError("Failed to find start frame in reference audio.");
                                }
                                g_findStartProcessing = false;
                            } })
                            .detach();
                    }
                    else
                    {
                        displayError("No reference audio selected.");
                    }
                }
            }
        }

        ImGui::SliderFloat("Size of start frame", &g_findStartSize, 1.0f, 10000.0f, "%1.0f");

        if (g_findStartFrame >= 0)
        {
            if (g_syncStartProcessing)
            {
                ImGui::Button("Processing...");
            }
            else
            {
                if (g_syncStartTime > 0.0)
                {
                    if (ImGui::Button(("Processed in " + std::to_string(g_syncStartTime) + " sec (click to reset)").c_str()))
                    {
                        g_syncStartTime = 0.0f;
                    }
                }
                else
                {
                    if (ImGui::Button("Sync start of all audios to ref"))
                    {
                        g_framesStartSync.clear();
                        int center = g_findStartFrame;
                        int radius = g_findStartSize;

                        int i = std::max(0, center - radius);
                        int j = std::min((int)audios[g_refAudioIndex].totalSampleCount, center + radius);

                        std::vector<float> refstart(
                            audios[g_refAudioIndex].samples + i,
                            audios[g_refAudioIndex].samples + j);

                        g_syncStartProcessing = true;
                        std::thread([&, refstart]()
                                    {
                            auto start = std::chrono::high_resolution_clock::now();
                            std::vector<int> localFramesStartSync;
                            for (size_t i = 0; i < audios.size(); ++i)
                            {
                                if (i == g_refAudioIndex)
                                {
                                    localFramesStartSync.push_back(-1);
                                    continue;
                                }
                                std::vector<float> signal(audios[i].samples, audios[i].samples + audios[i].totalSampleCount);
                                int position = Processing::findReferencePosition(signal, refstart);
                                localFramesStartSync.push_back(position);
                            }

                            auto end = std::chrono::high_resolution_clock::now();
                            std::chrono::duration<double> elapsed = end - start;
                            g_syncStartTime = elapsed.count();

                            {
                                std::lock_guard<std::mutex> lock(g_audioMutex);
                                g_framesStartSync = localFramesStartSync;
                                g_syncStartProcessing = false;
                            } })
                            .detach();
                    }
                }
            }
        }

        if (g_framesStartSync.size() == audios.size())
        {
            if (g_cutExcessProcessing)
            {
                ImGui::Button("Processing...");
            }
            else
            {
                if (g_cutExcessTime > 0.0)
                {
                    if (ImGui::Button(("Processed in " + std::to_string(g_cutExcessTime) + " sec (click to reset)").c_str()))
                    {
                        g_cutExcessTime = 0.0;
                    }
                }
                else
                {
                    if (ImGui::Button("Cut excess length"))
                    {
                        g_cutExcessProcessing = true;
                        std::thread([&]()
                                    {
                            auto start = std::chrono::high_resolution_clock::now();
                            for (size_t i = 0; i < audios.size(); ++i)
                            {
                                if (i == g_refAudioIndex)
                                {
                                    continue;
                                }
                                int diff = g_framesStartSync[i] - (g_findStartFrame - g_findStartSize);

                                if (diff > 0)
                                {
                                    drwav_uint64 newSampleCount = audios[i].totalSampleCount - diff * audios[i].channels;
                                    if (newSampleCount > 0)
                                    {
                                        float *newSamples = (float *)malloc(newSampleCount * sizeof(float));
                                        if (newSamples)
                                        {
                                            for (drwav_uint64 j = 0; j < newSampleCount; ++j)
                                            {
                                                newSamples[j] = audios[i].samples[j + diff * audios[i].channels];
                                            }
                                            drwav_free(audios[i].samples, NULL);
                                            audios[i].samples = newSamples;
                                            audios[i].totalSampleCount = newSampleCount * audios[i].channels;
                                        }
                                        else
                                        {
                                            displayError("Failed to allocate memory for trimmed audio samples.");
                                        }
                                    }
                                }
                                else
                                {
                                    drwav_uint64 padSamples = (-diff) * audios[i].channels;
                                    drwav_uint64 oldSamples = audios[i].totalSampleCount;
                                    drwav_uint64 newSampleCount = oldSamples + padSamples;

                                    float *newSamples = (float *)malloc(newSampleCount * sizeof(float));
                                    if (newSamples)
                                    {
                                        for (drwav_uint64 j = 0; j < padSamples; ++j)
                                            newSamples[j] = 0.0f;

                                        for (drwav_uint64 j = 0; j < oldSamples; ++j)
                                            newSamples[j + padSamples] = audios[i].samples[j];

                                        drwav_free(audios[i].samples, NULL);
                                        audios[i].samples = newSamples;
                                        audios[i].totalSampleCount = newSampleCount;
                                    }
                                }
                            }
                            auto end = std::chrono::high_resolution_clock::now();
                            std::chrono::duration<double> elapsed = end - start;
                            g_cutExcessTime = elapsed.count();

                            {
                                std::lock_guard<std::mutex> lock(g_audioMutex);
                                g_framesStartSync.clear();
                                g_findStartFrame = -1;
                                g_cutExcessProcessing = false;
                            } })
                            .detach();
                    }
                }
            }
        }

        ImGui::EndChild();
        ImGui::SameLine();
        ImGui::BeginChild("#DTW", ImVec2(io.DisplaySize.x / 3, 200));
        ImGui::Text("DTW");

        if (g_dtwProcessing)
        {
            ImGui::Button("Processing...");
        }
        else
        {
            if (g_dtwTime > 0.0)
            {
                if (ImGui::Button(("Processed in " + std::to_string(g_dtwTime) + " sec (click to reset)").c_str()))
                {
                    g_dtwTime = 0.0f;
                }
            }
            else
            {
                if (ImGui::Button("Sync all audios to reference"))
                {
                    g_dtwProcessing = true;
                    std::thread([&]()
                                {
                        auto start = std::chrono::high_resolution_clock::now();
                        std::vector<std::thread> threads;
                        for (size_t i = 0; i < audios.size(); ++i)
                        {
                            if (i == g_refAudioIndex)
                                continue;

                            threads.emplace_back([i, &audios]() {
                                std::vector<float> ref(audios[g_refAudioIndex].samples,
                                                    audios[g_refAudioIndex].samples + audios[g_refAudioIndex].totalSampleCount);
                                std::vector<float> signal(audios[i].samples,
                                                        audios[i].samples + audios[i].totalSampleCount);
                                std::vector<float> aligned = Processing::alignSignalToRef(ref, signal);

                                float *newSamples = (float *)malloc(aligned.size() * sizeof(float));
                                if (!newSamples)
                                {
                                    displayError("Failed to allocate memory for DTW aligned audio samples.");
                                    return;
                                }

                                for (size_t j = 0; j < aligned.size(); ++j)
                                {
                                    newSamples[j] = aligned[j];
                                }

                                {
                                    std::lock_guard<std::mutex> lock(g_audioMutex);
                                    drwav_free(audios[i].samples, nullptr);
                                    audios[i].samples = newSamples;
                                    audios[i].totalSampleCount = aligned.size() * audios[i].channels;
                                }
                            });
                        }

                        for (auto &t : threads)
                        {
                            if (t.joinable())
                                t.join();
                        }

                        auto end = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double> elapsed = end - start;
                        g_dtwTime = elapsed.count();
                        g_dtwProcessing = false; })
                        .detach();
                }
            }
        }

        ImGui::EndChild();

        int sampleRate = audios.empty() ? 44100 : audios[0].sampleRate;
        DrawTimelineScale(io.DisplaySize.x, 20, sampleRate, maxFrames);

        for (size_t i = 0; i < audios.size(); ++i)
        {
            ImGui::Text("%s", audios[i].name.c_str());
            if (i == g_refAudioIndex)
            {
                ImGui::Text("Im the ref");
            }
            else
            {
                if (ImGui::Button(("Define " + audios[i].name + " as reference audio").c_str()))
                {
                    g_refAudioIndex = i;
                    g_findStartFrame = -1;
                    g_framesStartSync.clear();
                }
            }
            ImGui::SameLine();
            if (ImGui::Button(("Remove " + audios[i].name).c_str()))
            {
                audios.erase(audios.begin() + i);
            }
            ImGui::SameLine();
            if (ImGui::Button(("Save " + audios[i].name).c_str()))
            {
                saveWavFile(audios[i]);
            }
            ImGui::BeginChild(("##audio" + std::to_string(i)).c_str(), ImVec2(io.DisplaySize.x, 80), false);
            DrawAudioOnTimeline(audios[i], io.DisplaySize.x, 80, maxFrames, g_refAudioIndex == i ? IM_COL32(0, 255, 0, 255) : IM_COL32(0, 0, 255, 255));

            if (g_isPlaying)
            {
                DrawProgressBar(io.DisplaySize.x, 80, maxFrames, sampleRate, audios);
            }
            if (i == g_refAudioIndex && g_findStartFrame >= 0)
            {
                DrawInitialStart(io.DisplaySize.x, 80, maxFrames, sampleRate, g_findStartFrame, g_findStartSize);
            }
            if (i != g_refAudioIndex && g_framesStartSync.size() > i)
            {
                DrawInitialStart(io.DisplaySize.x, 80, maxFrames, sampleRate, g_framesStartSync[i]);
            }

            if (!g_correlationResults.empty() && i < g_correlationResults.size())
            {
                DrawCorrelationMarkers(io.DisplaySize.x, 80, maxFrames, {g_correlationResults[i]});
            }
            ImGui::EndChild();
        }

        ImGui::End();

        if (g_errorPopup != "")
        {
            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x, io.DisplaySize.y));
            ImGui::Begin("##Error", NULL, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar);
            ImVec2 windowSize = ImGui::GetWindowSize();
            ImVec2 textSize = ImGui::CalcTextSize(g_errorPopup.c_str());

            ImVec2 textPos = {
                (windowSize.x - textSize.x) * 0.5f,
                (windowSize.y - textSize.y) * 0.5f};

            ImGui::SetCursorPos(textPos);
            ImGui::Text("%s", g_errorPopup.c_str());

            ImGui::End();

            if (ImGui::IsMouseClicked(0))
            {
                g_errorPopupTimer = 100;
            }

            if (g_errorPopupTimer < 100)
            {
                g_errorPopupTimer++;
            }
            else
            {
                g_errorPopup = "";
            }
        }

        fileDialog.Display();

        if (fileDialog.HasSelected())
        {
            auto file = fileDialog.GetSelected();
            AudioFile audio = readAudioFile(file.string(), file.filename().string());
            audios.push_back(audio);
            fileDialog.ClearSelected();
            maxFrames = GetLongestDurationFrames(audios);
        }

        ImGui::Render();
        glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
        glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        SDL_GL_SwapWindow(window);
    }

    if (g_audioDevice != 0)
        SDL_CloseAudioDevice(g_audioDevice);

    for (auto &file : audios)
    {
        drwav_free(file.samples, NULL);
    }

    return 0;
}