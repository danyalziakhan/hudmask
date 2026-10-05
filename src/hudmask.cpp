// HUD Mask
//
// Finds the game's HUD by its pixel shaders (CRC-32 of the bytecode, listed in
// hudmask.cfg) and binds it for effects as the texture semantic HUDMASK. Alpha
// is HUD coverage. Effects that don't read it are untouched.
//
// Two cases. If the HUD goes into a texture of its own, that texture is the
// mask. If it's drawn straight onto the back buffer, the frame is copied before
// the first HUD draw and diffed against the finished frame at present.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3dcompiler.h>
#define ImTextureID ImU64
#include <imgui.h>
#include <reshade.hpp>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace reshade::api;

extern "C" __declspec(dllexport) const char *NAME = "HUD Mask";
extern "C" __declspec(dllexport) const char *DESCRIPTION =
    "Finds the game's HUD by its shaders and gives it to ReShade effects as the HUDMASK texture.";

static const char *SECTION = "HUDMASK";
static const char *SEMANTIC = "HUDMASK";

// The shader list is per-game data rather than a user setting. It lives beside
// the add-on and ships with presets. The finder writes to it too.
static const wchar_t *CFG_NAME = L"hudmask.cfg";
static const char *CFG_KEY = "HudShaders";

//----------------------------------------------------------------------------

static uint32_t crc32(const uint8_t *data, size_t size)
{
    static uint32_t table[256] = {};
    static std::once_flag once;
    std::call_once(once, [] {
        for (uint32_t i = 0; i < 256; ++i)
        {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k)
                c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
    });
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < size; ++i)
        crc = (crc >> 8) ^ table[(crc ^ data[i]) & 0xFF];
    return ~crc;
}

//----------------------------------------------------------------------------

static std::atomic<bool> g_enabled{true};
// Debug=1 shows the shader list and the finder. Hidden otherwise: a wrong hash
// looks fine until it quietly masks the wrong thing.
static bool g_debug = false;
static HMODULE g_module = nullptr;
static std::wstring g_cfg_path;
static std::string g_hud_list;
static std::unordered_set<uint32_t> g_hud_hashes;

// Pixel shader handle -> hash. Created on loader threads, bound on render
// threads. The lock covers g_hud_hashes as well.
static std::shared_mutex g_shaders_lock;
static std::unordered_map<uint64_t, uint32_t> g_shaders;

// Per command list, because each D3D11 deferred context records on its own thread.
struct __declspec(uuid("5c8e2d1a-7b43-4e96-a1f0-3d2b9c64e817")) list_state
{
    uint64_t ps = 0;
    uint32_t hash = 0;
    bool hud = false;
    resource_view rtv = {0};
};

static list_state *state_of(command_list *cmd)
{
    if (list_state *s = cmd->get_private_data<list_state>())
        return s;
    return cmd->create_private_data<list_state>();
}

static effect_runtime *g_runtime = nullptr;

// HUD texture seen this frame, if the game uses one.
static std::atomic<uint64_t> g_frame_hud{0};
// Largest one wins. Witcher 3 has a HUD shader that also renders a 64x64 icon
// for the minimap; taking the last target would sometimes stretch that over the
// whole screen.
static std::mutex g_frame_hud_lock;
static uint64_t g_frame_hud_area = 0;

// SRVs for HUD textures. Odyssey ping-pongs between two. Each view holds a
// reference, so keep only a handful.
struct hud_view { uint64_t resource; resource_view srv; };
static std::vector<hud_view> g_views;
static uint64_t g_bound = 0;

// Games don't always redraw the HUD texture every frame (Odyssey's title screen
// skips some). Unbinding on those frames made the effects flicker under the
// text, so the last texture is held for ~0.5 s at 60 fps.
static const uint32_t HOLD_FRAMES = 30;
static uint64_t g_last_hud = 0;
static uint32_t g_frames_without = HOLD_FRAMES;

//----------------------------------------------------------------------------
// Mask built for HUDs drawn onto the back buffer. D3D11/D3D12 only: the diff
// shader is DXBC.

struct frame_mask
{
    resource_desc desc = {};
    resource clean = {0}, final_frame = {0}, mask = {0};
    resource_view clean_srv = {0}, final_srv = {0}, mask_rtv = {0}, mask_srv = {0};
};
static frame_mask g_fm;
static pipeline_layout g_fm_layout = {0};
static pipeline g_fm_pipeline = {0};
static bool g_fm_failed = false;
static std::atomic<bool> g_fm_ready{false};
// Set once a HUD draw hits the back buffer; resources are created lazily.
static std::atomic<bool> g_fm_wanted{false};
// Clean (pre-HUD) copy taken this frame.
static std::atomic<bool> g_clean_copied{false};
// Run the diff in the next reshade_begin_effects.
static bool g_build_now = false;

// Diff relative to the brighter pixel, so one threshold works for both 8-bit
// and scRGB float back buffers.
static const char *FM_SHADER = R"(
Texture2D<float4> t_clean : register(t0);
Texture2D<float4> t_final : register(t1);

void vs_main(uint id : SV_VertexID, out float4 pos : SV_Position)
{
    float2 uv = float2(id == 1 ? 2.0 : 0.0, id == 2 ? 2.0 : 0.0);
    pos = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

float4 ps_main(float4 pos : SV_Position) : SV_Target
{
    int3 p = int3(pos.xy, 0);
    float3 a = t_clean.Load(p).rgb;
    float3 b = t_final.Load(p).rgb;
    float3 d = abs(a - b);
    float peak = max(max(max(a.r, a.g), max(a.b, b.r)), max(b.g, b.b));
    float diff = max(max(d.r, d.g), d.b) / max(peak, 1.0);
    return float4(0.0, 0.0, 0.0, smoothstep(0.5 / 255.0, 2.0 / 255.0, diff));
}
)";

// Panel stats, 120-frame window.
static uint32_t g_window_frames = 0, g_window_texture = 0, g_window_built = 0;
static float g_share_texture = 0.0f, g_share_built = 0.0f;
static resource_desc g_last_desc = {};

//----------------------------------------------------------------------------
// Finder (debug only). A scan collects every pixel shader drawing into a
// screen-sized target, ordered by how late in the frame it draws. Stepping
// through makes one shader blink at a time. Blinking beats hiding here: you see
// the actual element in place, and it works for both HUD paths.

struct candidate
{
    uint32_t hash = 0;
    uint32_t frames = 0;
    uint32_t draws = 0;
    double position = 0.0;  // summed over frames, 0 first draw to 1 last
    bool on_screen = false;
};

static std::mutex g_finder_lock;
static int g_scan_frames = 0;
static std::unordered_map<uint32_t, candidate> g_scan;
static std::unordered_map<uint32_t, uint32_t> g_scan_frame_pos;
static std::atomic<uint32_t> g_scan_counter{0};
static std::vector<candidate> g_found;
static int g_step = -1;
static std::atomic<uint32_t> g_blinking{0};
static std::atomic<bool> g_blink_list{false};
static uint32_t g_bb_width = 0, g_bb_height = 0;

// "Log one frame": records (target, shader) for every draw in one frame, then
// reports the HUD path and any unlisted shaders drawing where the HUD goes.
struct log_key
{
    uint64_t target;
    uint32_t hash;
    bool on_screen;
    bool operator<(const log_key &o) const
    {
        return target != o.target ? target < o.target : hash != o.hash ? hash < o.hash : on_screen < o.on_screen;
    }
};
struct log_entry { uint32_t draws = 0, first = 0; };
static std::atomic<int> g_log_state{0};  // 1 armed, 2 recording this frame
static std::mutex g_log_lock;
static std::map<log_key, log_entry> g_log;
static std::atomic<uint32_t> g_log_order{0};
static std::vector<std::string> g_log_lines;
struct suggestion { uint32_t hash, draws; };
static std::vector<suggestion> g_suggest;

// 2 Hz blink: draws are skipped during the off half.
static bool blink_off()
{
    return (GetTickCount64() / 250) % 2 == 0;
}

//----------------------------------------------------------------------------

static void parse_hud_list()
{
    g_hud_hashes.clear();
    const char *p = g_hud_list.c_str();
    while (*p)
    {
        char *end = nullptr;
        const unsigned long v = strtoul(p, &end, 10);
        if (end == p)
        {
            ++p;
            continue;
        }
        if (v != 0)
            g_hud_hashes.insert(static_cast<uint32_t>(v));
        p = end;
    }
}

static std::vector<std::string> read_cfg_lines()
{
    std::vector<std::string> lines;
    if (FILE *f = _wfopen(g_cfg_path.c_str(), L"r"))
    {
        char line[4096];
        while (fgets(line, sizeof(line), f))
        {
            std::string l = line;
            while (!l.empty() && (l.back() == '\n' || l.back() == '\r'))
                l.pop_back();
            lines.push_back(l);
        }
        fclose(f);
    }
    return lines;
}

static bool is_key_line(const std::string &l)
{
    const size_t start = l.find_first_not_of(" \t");
    return start != std::string::npos && l.compare(start, strlen(CFG_KEY), CFG_KEY) == 0 &&
           l.find('=', start) != std::string::npos;
}

// Only the HudShaders line is rewritten. Comments in the file are kept.
static void save_hud_list()
{
    std::string list;
    {
        std::shared_lock lock(g_shaders_lock);
        std::vector<uint32_t> sorted(g_hud_hashes.begin(), g_hud_hashes.end());
        std::sort(sorted.begin(), sorted.end());
        for (uint32_t h : sorted)
            list += (list.empty() ? "" : ",") + std::to_string(h);
    }
    g_hud_list = list;

    std::vector<std::string> lines = read_cfg_lines();
    if (lines.empty())
        lines.push_back("# HUD Mask: the pixel shaders that draw this game's HUD, as CRC-32 of their bytecode.");
    const std::string key_line = std::string(CFG_KEY) + "=" + list;
    bool replaced = false;
    for (std::string &l : lines)
        if (is_key_line(l))
        {
            l = key_line;
            replaced = true;
        }
    if (!replaced)
        lines.push_back(key_line);

    if (FILE *f = _wfopen(g_cfg_path.c_str(), L"w"))
    {
        for (const std::string &l : lines)
            fprintf(f, "%s\n", l.c_str());
        fclose(f);
    }
}

static void load_settings()
{
    bool enabled = true;
    reshade::get_config_value(nullptr, SECTION, "Enabled", enabled);
    g_enabled = enabled;
    reshade::get_config_value(nullptr, SECTION, "Debug", g_debug);

    wchar_t path[MAX_PATH] = L"";
    GetModuleFileNameW(g_module, path, MAX_PATH);
    g_cfg_path = path;
    g_cfg_path = g_cfg_path.substr(0, g_cfg_path.find_last_of(L"\\/") + 1) + CFG_NAME;

    for (const std::string &l : read_cfg_lines())
        if (is_key_line(l))
            g_hud_list = l.substr(l.find('=') + 1);
    parse_hud_list();
}

//----------------------------------------------------------------------------

static void on_init_pipeline(device *, pipeline_layout, uint32_t count, const pipeline_subobject *subobjects, pipeline p)
{
    for (uint32_t i = 0; i < count; ++i)
    {
        if (subobjects[i].type != pipeline_subobject_type::pixel_shader || subobjects[i].data == nullptr)
            continue;
        const auto &desc = *static_cast<const shader_desc *>(subobjects[i].data);
        if (desc.code == nullptr || desc.code_size == 0)
            continue;
        const uint32_t hash = crc32(static_cast<const uint8_t *>(desc.code), desc.code_size);
        std::unique_lock lock(g_shaders_lock);
        g_shaders[p.handle] = hash;
    }
}

static void on_destroy_pipeline(device *, pipeline p)
{
    std::unique_lock lock(g_shaders_lock);
    g_shaders.erase(p.handle);
}

static void on_bind_pipeline(command_list *cmd, pipeline_stage stages, pipeline p)
{
    if ((stages & pipeline_stage::pixel_shader) != pipeline_stage::pixel_shader)
        return;
    list_state *const st = state_of(cmd);
    if (p.handle == st->ps)
        return;
    st->ps = p.handle;
    st->hash = 0;
    st->hud = false;
    if (p.handle != 0)
    {
        std::shared_lock lock(g_shaders_lock);
        if (auto it = g_shaders.find(p.handle); it != g_shaders.end())
        {
            st->hash = it->second;
            st->hud = g_hud_hashes.count(it->second) != 0;
        }
    }
}

static void on_bind_render_targets(command_list *cmd, uint32_t count, const resource_view *rtvs, resource_view)
{
    state_of(cmd)->rtv = count != 0 ? rtvs[0] : resource_view{0};
}

// Deferred contexts and D3D12 lists lose their bindings on reset.
static void on_reset_command_list(command_list *cmd)
{
    *state_of(cmd) = list_state();
}

static void on_destroy_command_list(command_list *cmd)
{
    if (cmd->get_private_data<list_state>() != nullptr)
        cmd->destroy_private_data<list_state>();
}

//----------------------------------------------------------------------------

static void finder_record(command_list *cmd, const list_state *st, resource target, bool on_screen)
{
    if (st->hash == 0 || target.handle == 0)
        return;
    const resource_desc d = cmd->get_device()->get_resource_desc(target);
    if (d.texture.width != g_bb_width || d.texture.height != g_bb_height)
        return;
    const uint32_t pos = ++g_scan_counter;
    std::lock_guard lock(g_finder_lock);
    candidate &c = g_scan[st->hash];
    c.hash = st->hash;
    ++c.draws;
    c.on_screen |= on_screen;
    uint32_t &last = g_scan_frame_pos[st->hash];
    last = std::max(last, pos);
}

// Returning true skips the draw (finder blink only).
static bool on_any_draw(command_list *cmd)
{
    const list_state *const st = cmd->get_private_data<list_state>();
    if (st == nullptr || g_runtime == nullptr)
        return false;

    const bool scanning = g_scan_frames > 0;
    const bool logging = g_log_state == 2;
    if (!scanning && !logging && !st->hud)
        return st->hash != 0 && st->hash == g_blinking && blink_off();
    if (st->rtv.handle == 0)
        return false;

    device *const dev = cmd->get_device();
    const resource target = dev->get_resource_from_view(st->rtv);
    const resource back_buffer = g_runtime->get_current_back_buffer();

    if (scanning)
        finder_record(cmd, st, target, target == back_buffer);
    if (logging && st->hash != 0)
    {
        const uint32_t order = ++g_log_order;
        std::lock_guard lock(g_log_lock);
        log_entry &e = g_log[{target.handle, st->hash, target == back_buffer}];
        if (e.draws++ == 0)
            e.first = order;
    }
    if (((st->hash != 0 && st->hash == g_blinking) || (st->hud && g_blink_list)) && blink_off())
        return true;
    if (!st->hud || !g_enabled)
        return false;

    if (target != back_buffer)
    {
        const resource_desc d = dev->get_resource_desc(target);
        const uint64_t area = uint64_t(d.texture.width) * d.texture.height;
        std::lock_guard lock(g_frame_hud_lock);
        if (area > g_frame_hud_area || g_frame_hud == 0)
        {
            g_frame_hud_area = area;
            g_frame_hud = target.handle;
        }
        return false;
    }

    // HUD drawn onto the back buffer: grab the frame before the first HUD draw.
    g_fm_wanted = true;
    if (!g_fm_ready || g_clean_copied.exchange(true))
        return false;
    cmd->barrier(back_buffer, resource_usage::render_target, resource_usage::copy_source);
    cmd->copy_resource(back_buffer, g_fm.clean);
    cmd->barrier(back_buffer, resource_usage::copy_source, resource_usage::render_target);
    return false;
}

static bool on_draw(command_list *cmd, uint32_t, uint32_t, uint32_t, uint32_t)
{
    return on_any_draw(cmd);
}

static bool on_draw_indexed(command_list *cmd, uint32_t, uint32_t, uint32_t, int32_t, uint32_t)
{
    return on_any_draw(cmd);
}

static bool on_draw_indirect(command_list *cmd, indirect_command type, resource, uint64_t, uint32_t, uint32_t)
{
    return type != indirect_command::dispatch && on_any_draw(cmd);
}

//----------------------------------------------------------------------------

static bool compile(const char *entry, const char *target, std::vector<uint8_t> &out)
{
    static HMODULE compiler = LoadLibraryW(L"d3dcompiler_47.dll");
    const auto compile_fn = compiler != nullptr ? reinterpret_cast<pD3DCompile>(GetProcAddress(compiler, "D3DCompile")) : nullptr;
    if (compile_fn == nullptr)
        return false;
    ID3DBlob *code = nullptr, *errors = nullptr;
    const HRESULT hr = compile_fn(FM_SHADER, strlen(FM_SHADER), nullptr, nullptr, nullptr, entry, target,
                                  D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors);
    if (errors != nullptr)
    {
        reshade::log::message(reshade::log::level::error, static_cast<const char *>(errors->GetBufferPointer()));
        errors->Release();
    }
    if (FAILED(hr) || code == nullptr)
        return false;
    const auto *bytes = static_cast<const uint8_t *>(code->GetBufferPointer());
    out.assign(bytes, bytes + code->GetBufferSize());
    code->Release();
    return true;
}

static bool create_pipeline(device *dev)
{
    std::vector<uint8_t> vs, ps;
    if (!compile("vs_main", "vs_5_0", vs) || !compile("ps_main", "ps_5_0", ps))
        return false;

    const pipeline_layout_param param = descriptor_range{0, 0, 0, 2, shader_stage::pixel, 1, descriptor_type::shader_resource_view};
    if (!dev->create_pipeline_layout(1, &param, &g_fm_layout))
        return false;

    shader_desc vs_desc = {vs.data(), vs.size()};
    shader_desc ps_desc = {ps.data(), ps.size()};
    format rt_format = format::r8g8b8a8_unorm;
    primitive_topology topology = primitive_topology::triangle_list;
    rasterizer_desc rasterizer;
    rasterizer.cull_mode = cull_mode::none;
    depth_stencil_desc depth;
    depth.depth_enable = false;
    pipeline_subobject subobjects[] = {
        {pipeline_subobject_type::vertex_shader, 1, &vs_desc},
        {pipeline_subobject_type::pixel_shader, 1, &ps_desc},
        {pipeline_subobject_type::render_target_formats, 1, &rt_format},
        {pipeline_subobject_type::primitive_topology, 1, &topology},
        {pipeline_subobject_type::rasterizer_state, 1, &rasterizer},
        {pipeline_subobject_type::depth_stencil_state, 1, &depth},
    };
    return dev->create_pipeline(g_fm_layout, static_cast<uint32_t>(std::size(subobjects)), subobjects, &g_fm_pipeline);
}

static void release_frame_mask(device *dev)
{
    g_fm_ready = false;
    for (resource_view v : {g_fm.clean_srv, g_fm.final_srv, g_fm.mask_rtv, g_fm.mask_srv})
        if (v.handle != 0)
            dev->destroy_resource_view(v);
    for (resource r : {g_fm.clean, g_fm.final_frame, g_fm.mask})
        if (r.handle != 0)
            dev->destroy_resource(r);
    g_fm = frame_mask();
}

static bool create_frame_mask(device *dev, const resource_desc &bb)
{
    if (g_fm_failed)
        return false;
    const device_api api = dev->get_api();
    if (api != device_api::d3d11 && api != device_api::d3d12)
    {
        g_fm_failed = true;
        return false;
    }
    if (g_fm_pipeline.handle == 0 && !create_pipeline(dev))
    {
        reshade::log::message(reshade::log::level::error, "HUD Mask could not build its comparison pass");
        g_fm_failed = true;
        return false;
    }

    frame_mask fm;
    fm.desc = bb;
    const resource_desc copy_desc(bb.texture.width, bb.texture.height, 1, 1, format_to_typeless(bb.texture.format), 1,
                                  memory_heap::default_, resource_usage::copy_dest | resource_usage::shader_resource);
    const resource_desc mask_desc(bb.texture.width, bb.texture.height, 1, 1, format::r8g8b8a8_unorm, 1,
                                  memory_heap::default_, resource_usage::render_target | resource_usage::shader_resource);
    const resource_view_desc copy_view(format_to_default_typed(bb.texture.format, 0));
    const bool ok =
        dev->create_resource(copy_desc, nullptr, resource_usage::copy_dest, &fm.clean) &&
        dev->create_resource(copy_desc, nullptr, resource_usage::copy_dest, &fm.final_frame) &&
        dev->create_resource(mask_desc, nullptr, resource_usage::shader_resource, &fm.mask) &&
        dev->create_resource_view(fm.clean, resource_usage::shader_resource, copy_view, &fm.clean_srv) &&
        dev->create_resource_view(fm.final_frame, resource_usage::shader_resource, copy_view, &fm.final_srv) &&
        dev->create_resource_view(fm.mask, resource_usage::render_target, resource_view_desc(format::r8g8b8a8_unorm), &fm.mask_rtv) &&
        dev->create_resource_view(fm.mask, resource_usage::shader_resource, resource_view_desc(format::r8g8b8a8_unorm), &fm.mask_srv);
    g_fm = fm;
    if (!ok)
    {
        release_frame_mask(dev);
        g_fm_failed = true;
        return false;
    }
    g_fm_ready = true;
    return true;
}

// Runs inside ReShade's effect pass, which already saves and restores the
// game's state around it. Free to bind anything here.
static void on_begin_effects(effect_runtime *runtime, command_list *cmd, resource_view, resource_view)
{
    if (!g_build_now)
        return;
    g_build_now = false;

    const resource back_buffer = runtime->get_current_back_buffer();
    cmd->barrier(back_buffer, resource_usage::render_target, resource_usage::copy_source);
    cmd->copy_resource(back_buffer, g_fm.final_frame);
    cmd->barrier(back_buffer, resource_usage::copy_source, resource_usage::render_target);

    const resource copies[2] = {g_fm.clean, g_fm.final_frame};
    const resource_usage copy_dest[2] = {resource_usage::copy_dest, resource_usage::copy_dest};
    const resource_usage readable[2] = {resource_usage::shader_resource, resource_usage::shader_resource};
    cmd->barrier(2, copies, copy_dest, readable);
    cmd->barrier(g_fm.mask, resource_usage::shader_resource, resource_usage::render_target);

    cmd->bind_pipeline(pipeline_stage::all_graphics, g_fm_pipeline);
    const resource_view srvs[2] = {g_fm.clean_srv, g_fm.final_srv};
    cmd->push_descriptors(shader_stage::pixel, g_fm_layout, 0,
                          descriptor_table_update{{}, 0, 0, 2, descriptor_type::shader_resource_view, srvs});
    const viewport vp = {0.0f, 0.0f, float(g_fm.desc.texture.width), float(g_fm.desc.texture.height), 0.0f, 1.0f};
    cmd->bind_viewports(0, 1, &vp);
    const rect scissor = {0, 0, int32_t(g_fm.desc.texture.width), int32_t(g_fm.desc.texture.height)};
    cmd->bind_scissor_rects(0, 1, &scissor);
    cmd->bind_render_targets_and_depth_stencil(1, &g_fm.mask_rtv);
    cmd->draw(3, 1, 0, 0);

    cmd->barrier(g_fm.mask, resource_usage::render_target, resource_usage::shader_resource);
    cmd->barrier(2, copies, readable, copy_dest);
}

//----------------------------------------------------------------------------

static void bind(uint64_t resource_handle, resource_view srv)
{
    if (g_runtime != nullptr)
        g_runtime->update_texture_bindings(SEMANTIC, srv, srv);
    g_bound = resource_handle;
}

static resource_view view_of(device *dev, resource res)
{
    for (const hud_view &v : g_views)
        if (v.resource == res.handle)
            return v.srv;

    const resource_desc desc = dev->get_resource_desc(res);
    resource_view srv = {0};
    if (!dev->create_resource_view(res, resource_usage::shader_resource,
            resource_view_desc(format_to_default_typed(desc.texture.format, 0)), &srv))
        return {0};
    g_last_desc = desc;

    if (g_views.size() >= 4)
    {
        for (auto it = g_views.begin(); it != g_views.end(); ++it)
            if (it->resource != g_bound)
            {
                dev->destroy_resource_view(it->srv);
                g_views.erase(it);
                break;
            }
    }
    g_views.push_back({res.handle, srv});
    return srv;
}

static void release_views(device *dev)
{
    if (g_bound != 0)
        bind(0, {0});
    for (const hud_view &v : g_views)
        dev->destroy_resource_view(v.srv);
    g_views.clear();
    release_frame_mask(dev);
}

static void set_step(int step)
{
    g_step = g_found.empty() ? -1 : std::clamp(step, -1, int(g_found.size()) - 1);
    g_blinking = g_step >= 0 ? g_found[g_step].hash : 0u;
}

static void toggle_in_list(uint32_t hash)
{
    {
        std::unique_lock lock(g_shaders_lock);
        if (!g_hud_hashes.erase(hash))
            g_hud_hashes.insert(hash);
    }
    save_hud_list();
}

static void finder_end_of_frame()
{
    if (g_scan_frames <= 0)
        return;
    std::lock_guard lock(g_finder_lock);
    const double total = std::max<uint32_t>(g_scan_counter.exchange(0), 1);
    for (const auto &[hash, pos] : g_scan_frame_pos)
    {
        candidate &c = g_scan[hash];
        c.position += pos / total;
        ++c.frames;
    }
    g_scan_frame_pos.clear();

    if (--g_scan_frames == 0)
    {
        // HUD tends to draw last, so put the latest first.
        g_found.clear();
        for (const auto &[hash, c] : g_scan)
            g_found.push_back(c);
        std::sort(g_found.begin(), g_found.end(), [](const candidate &a, const candidate &b) {
            return a.position / std::max(a.frames, 1u) > b.position / std::max(b.frames, 1u);
        });
        if (g_found.size() > 64)
            g_found.resize(64);
        g_scan.clear();
        set_step(-1);
    }
}

static void logf(const char *fmt, ...)
{
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    g_log_lines.push_back(buf);
    reshade::log::message(reshade::log::level::info, buf);
}

static const char *format_name(format f)
{
    switch (f)
    {
    case format::r8g8b8a8_unorm: return "RGBA8";
    case format::r8g8b8a8_unorm_srgb: return "RGBA8 sRGB";
    case format::b8g8r8a8_unorm: return "BGRA8";
    case format::b8g8r8a8_unorm_srgb: return "BGRA8 sRGB";
    case format::r10g10b10a2_unorm: return "RGB10A2";
    case format::r11g11b10_float: return "R11G11B10 float";
    case format::r16g16b16a16_float: return "RGBA16 float";
    default: return "other";
    }
}

static std::string describe(device *dev, uint64_t target)
{
    const resource_desc d = dev->get_resource_desc(resource{target});
    char buf[96];
    snprintf(buf, sizeof(buf), "%u x %u %s (format %u)", d.texture.width, d.texture.height,
             format_name(d.texture.format), static_cast<uint32_t>(d.texture.format));
    return buf;
}

static void analyse_log(device *dev)
{
    std::map<log_key, log_entry> draws;
    {
        std::lock_guard lock(g_log_lock);
        draws.swap(g_log);
    }
    std::unordered_set<uint32_t> listed;
    {
        std::shared_lock lock(g_shaders_lock);
        listed = g_hud_hashes;
    }
    g_log_lines.clear();
    g_suggest.clear();

    std::map<uint64_t, uint32_t> hud_targets;
    std::unordered_set<uint32_t> seen;
    uint32_t screen_draws = 0, first_on_screen = UINT32_MAX;
    for (const auto &[k, e] : draws)
    {
        if (listed.count(k.hash) == 0)
            continue;
        seen.insert(k.hash);
        if (k.on_screen)
        {
            screen_draws += e.draws;
            first_on_screen = std::min(first_on_screen, e.first);
        }
        else
        {
            hud_targets[k.target] += e.draws;
        }
    }
    uint64_t main_target = 0;
    uint32_t main_draws = 0;
    for (const auto &[t, n] : hud_targets)
        if (n > main_draws)
        {
            main_target = t;
            main_draws = n;
        }

    logf("HUD Mask, one frame: %zu shaders in the list, %zu of them drew", listed.size(), seen.size());
    if (seen.empty())
    {
        logf("No listed shader drew this frame. Is the HUD on screen, and is the list for this game?");
        return;
    }
    if (main_target != 0 && screen_draws == 0)
        logf("Path: HUD drawn into its own texture, %s, %u draws by listed shaders",
             describe(dev, main_target).c_str(), main_draws);
    else if (main_target == 0)
        logf("Path: HUD drawn onto the frame, %u draws by listed shaders", screen_draws);
    else
        logf("Path: both. %u draws onto the frame, and %u into a texture, %s. The frame wins.",
             screen_draws, main_draws, describe(dev, main_target).c_str());
    for (const auto &[t, n] : hud_targets)
        if (t != main_target)
            logf("Also: %u draws by listed shaders into %s. A listed shader may draw more than HUD.",
                 n, describe(dev, t).c_str());
    for (uint32_t h : listed)
        if (seen.count(h) == 0)
            logf("Not drawn this frame: %u", h);

    // Suggestions: unlisted shaders drawing into the HUD texture, or onto the
    // back buffer after the first HUD draw.
    std::map<uint32_t, uint32_t> found;
    for (const auto &[k, e] : draws)
    {
        if (listed.count(k.hash) != 0)
            continue;
        if ((main_target != 0 && k.target == main_target) || (k.on_screen && screen_draws != 0 && e.first > first_on_screen))
            found[k.hash] += e.draws;
    }
    for (const auto &[h, n] : found)
        g_suggest.push_back({h, n});
    if (g_suggest.empty())
        logf("No other shader draws where the HUD goes. The list looks complete for this frame.");
    else
        for (const suggestion &sg : g_suggest)
            logf("Likely HUD as well: %u, %u draws", sg.hash, sg.draws);

    std::string line;
    for (uint32_t h : seen)
        line += (line.empty() ? "" : ",") + std::to_string(h);
    logf("Listed shaders that drew: %s", line.c_str());
}

static void log_end_of_frame(device *dev)
{
    if (g_log_state == 2)
    {
        g_log_state = 0;
        analyse_log(dev);
    }
    else if (g_log_state == 1)
    {
        std::lock_guard lock(g_log_lock);
        g_log.clear();
        g_log_order = 0;
        g_log_state = 2;
    }
}

// Fires before ReShade renders effects for this frame.
static void on_present(command_queue *queue, swapchain *, const rect *, const rect *, uint32_t, const rect *)
{
    device *const dev = queue->get_device();
    uint64_t hud;
    {
        std::lock_guard lock(g_frame_hud_lock);
        hud = g_frame_hud.exchange(0);
        g_frame_hud_area = 0;
    }
    const bool copied = g_clean_copied.exchange(false);

    if (g_runtime != nullptr)
    {
        const resource_desc bb = dev->get_resource_desc(g_runtime->get_current_back_buffer());
        g_bb_width = bb.texture.width;
        g_bb_height = bb.texture.height;
        if (g_fm_wanted && bb.texture.samples == 1 &&
            (!g_fm_ready || g_fm.desc.texture.width != bb.texture.width || g_fm.desc.texture.height != bb.texture.height ||
             g_fm.desc.texture.format != bb.texture.format))
        {
            if (g_bound != 0 && g_bound == g_fm.mask.handle)
                bind(0, {0});
            release_frame_mask(dev);
            create_frame_mask(dev, bb);
        }
    }

    finder_end_of_frame();
    log_end_of_frame(dev);
    if (g_debug && g_runtime != nullptr && g_runtime->is_key_pressed(VK_DELETE))
        g_blink_list = !g_blink_list;
    if (g_debug && !g_found.empty() && g_runtime != nullptr)
    {
        if (g_runtime->is_key_pressed(VK_NEXT))
            set_step(g_step + 1);
        if (g_runtime->is_key_pressed(VK_PRIOR))
            set_step(g_step - 1);
        if (g_runtime->is_key_pressed(VK_END))
            set_step(-1);
        if (g_runtime->is_key_pressed(VK_HOME) && g_step >= 0)
            toggle_in_list(g_found[g_step].hash);
    }

    if (++g_window_frames >= 120)
    {
        g_share_texture = float(g_window_texture) / float(g_window_frames);
        g_share_built = float(g_window_built) / float(g_window_frames);
        g_window_frames = g_window_texture = g_window_built = 0;
    }
    if (hud != 0)
        ++g_window_texture;

    if (hud != 0)
    {
        g_last_hud = hud;
        g_frames_without = 0;
    }
    else if (g_frames_without < HOLD_FRAMES)
    {
        ++g_frames_without;
    }

    uint64_t want = 0;
    resource_view srv = {0};
    if (g_enabled && copied && g_fm_ready)
    {
        // No hold needed here. A back-buffer HUD is redrawn every frame.
        ++g_window_built;
        g_build_now = true;
        want = g_fm.mask.handle;
        srv = g_fm.mask_srv;
    }
    else if (g_enabled && g_frames_without < HOLD_FRAMES)
    {
        want = g_last_hud;
    }

    if (want == g_bound || g_runtime == nullptr)
        return;
    if (want == 0)
    {
        bind(0, {0});
        return;
    }
    if (srv.handle == 0)
        srv = view_of(dev, resource{want});
    bind(srv.handle != 0 ? want : 0, srv);
}

// For another add-on in the same process, such as HDR Bridge, which reads the
// HUD at present before this add-on's own callback may have run. Returns 1 and
// a view of this frame's HUD texture when the game draws its HUD into one; 2
// when it draws onto the back buffer, where the mask is only built after
// present and so cannot be offered here; 0 when there is no HUD or the add-on
// is off. dev is the reshade::api::device the two share. The view belongs to
// this add-on and is good for the current frame only.
extern "C" __declspec(dllexport) int hudmask_frame_texture(void *dev, uint64_t *srv)
{
    *srv = 0;
    if (!g_enabled || dev == nullptr)
        return 0;
    if (g_clean_copied || (g_bound != 0 && g_bound == g_fm.mask.handle))
        return 2;
    // Drawn this frame and not yet taken by on_present, or already bound by it.
    uint64_t hud = g_frame_hud.load();
    if (hud == 0)
        hud = g_bound;
    if (hud == 0)
        return 0;
    *srv = view_of(static_cast<device *>(dev), resource{hud}).handle;
    return *srv != 0 ? 1 : 0;
}

static void on_init_effect_runtime(effect_runtime *runtime)
{
    g_runtime = runtime;
}

static void on_destroy_effect_runtime(effect_runtime *runtime)
{
    if (runtime != g_runtime)
        return;
    device *const dev = runtime->get_device();
    release_views(dev);
    if (g_fm_pipeline.handle != 0)
        dev->destroy_pipeline(g_fm_pipeline);
    if (g_fm_layout.handle != 0)
        dev->destroy_pipeline_layout(g_fm_layout);
    g_fm_pipeline = {0};
    g_fm_layout = {0};
    g_runtime = nullptr;
}

static void on_destroy_swapchain(swapchain *sc, bool)
{
    release_views(sc->get_device());
}

//----------------------------------------------------------------------------

static void draw_finder()
{
    ImGui::Separator();
    ImGui::TextUnformatted("HUD finder");
    bool blink_list = g_blink_list;
    if (ImGui::Checkbox("Blink the whole list (Delete)", &blink_list))
        g_blink_list = blink_list;
    ImGui::SameLine();
    ImGui::TextDisabled("any part of the HUD that stays steady is missing");
    if (g_scan_frames > 0)
    {
        ImGui::Text("Scanning, %d frames left", g_scan_frames);
        return;
    }
    if (ImGui::Button("Scan"))
    {
        std::lock_guard lock(g_finder_lock);
        g_scan.clear();
        g_scan_frame_pos.clear();
        g_scan_counter = 0;
        g_found.clear();
        set_step(-1);
        g_scan_frames = 60;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("with the HUD on screen");
    if (g_found.empty())
        return;

    ImGui::TextWrapped("With the overlay closed: Page Down and Page Up make the next or previous shader blink, "
                       "Home adds the blinking one to the HUD list or takes it off, End stops it. Latest in the "
                       "frame is at the top, which is where a HUD draws.");

    if (ImGui::BeginTable("found", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit,
                          ImVec2(0.0f, 16.0f * ImGui::GetTextLineHeightWithSpacing())))
    {
        ImGui::TableSetupColumn("HUD");
        ImGui::TableSetupColumn("Shader");
        ImGui::TableSetupColumn("Draws");
        ImGui::TableSetupColumn("Target");
        ImGui::TableHeadersRow();
        for (int i = 0; i < int(g_found.size()); ++i)
        {
            const candidate &c = g_found[i];
            ImGui::PushID(i);
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            bool in_list;
            {
                std::shared_lock lock(g_shaders_lock);
                in_list = g_hud_hashes.count(c.hash) != 0;
            }
            if (ImGui::Checkbox("##in", &in_list))
                toggle_in_list(c.hash);
            ImGui::TableNextColumn();
            char label[32];
            snprintf(label, sizeof(label), "%u", c.hash);
            if (ImGui::Selectable(label, g_step == i, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap))
                set_step(g_step == i ? -1 : i);
            ImGui::TableNextColumn();
            ImGui::Text("%u", c.draws / std::max(c.frames, 1u));
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(c.on_screen ? "screen" : "texture");
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    if (g_step >= 0)
        ImGui::Text("Blinking: %u. Click it again or press End to stop.", g_found[g_step].hash);
}

static void draw_log()
{
    ImGui::Separator();
    if (g_log_state != 0)
        ImGui::TextUnformatted("Logging the next frame...");
    else if (ImGui::Button("Log one frame"))
        g_log_state = 1;
    ImGui::SameLine();
    ImGui::TextDisabled("how the HUD is drawn, and what else draws where it goes; also in ReShade.log");
    for (const std::string &l : g_log_lines)
        ImGui::TextWrapped("%s", l.c_str());
    if (g_suggest.empty())
        return;

    ImGui::TextDisabled("Click one to make it blink. Tick it to add it to the list.");
    for (size_t i = 0; i < g_suggest.size(); ++i)
    {
        const suggestion &sg = g_suggest[i];
        ImGui::PushID(int(i));
        bool in_list;
        {
            std::shared_lock lock(g_shaders_lock);
            in_list = g_hud_hashes.count(sg.hash) != 0;
        }
        if (ImGui::Checkbox("##add", &in_list))
            toggle_in_list(sg.hash);
        ImGui::SameLine();
        char label[48];
        snprintf(label, sizeof(label), "%u (%u draws)", sg.hash, sg.draws);
        if (ImGui::Selectable(label, g_blinking == sg.hash))
            g_blinking = g_blinking == sg.hash ? 0u : sg.hash;
        ImGui::PopID();
    }
    if (ImGui::Button("Add all of them"))
    {
        for (const suggestion &sg : g_suggest)
        {
            std::unique_lock lock(g_shaders_lock);
            g_hud_hashes.insert(sg.hash);
        }
        save_hud_list();
    }
}

static void draw_settings(effect_runtime *)
{
    bool enabled = g_enabled;
    if (ImGui::Checkbox("Give effects the HUD mask", &enabled))
    {
        g_enabled = enabled;
        reshade::set_config_value(nullptr, SECTION, "Enabled", enabled);
    }

    size_t known;
    {
        std::shared_lock lock(g_shaders_lock);
        known = g_hud_hashes.size();
    }
    if (known == 0)
        ImGui::TextWrapped("No hudmask.cfg with this game's HUD shaders beside the add-on. Set Debug=1 under "
                           "[HUDMASK] in ReShade.ini to find them.");
    else if (g_fm_failed)
        ImGui::TextWrapped("The HUD is drawn onto the frame, and building a mask from it is not supported with "
                           "this graphics API.");
    else if (g_share_built > 0.0f)
        ImGui::TextWrapped("HUD drawn onto the frame: mask built from the frame before and after it, in %.0f%% of "
                           "recent frames.", g_share_built * 100.0f);
    else
        ImGui::TextWrapped("HUD drawn into its own texture: its alpha is the mask, in %.0f%% of recent frames.",
                           g_share_texture * 100.0f);
    if (!g_debug)
        return;

    ImGui::Separator();
    // Count distinct hashes. Some games create the same shader twice.
    size_t created = 0;
    {
        std::shared_lock lock(g_shaders_lock);
        std::unordered_set<uint32_t> present;
        for (const auto &[handle, hash] : g_shaders)
            if (g_hud_hashes.count(hash) != 0)
                present.insert(hash);
        created = present.size();
    }
    ImGui::Text("HUD shaders the game has created: %zu of %zu", created, known);
    if (g_last_desc.texture.width != 0)
        ImGui::Text("HUD texture: %u x %u, format %u", g_last_desc.texture.width, g_last_desc.texture.height,
                    static_cast<uint32_t>(g_last_desc.texture.format));

    char buf[4096];
    strncpy(buf, g_hud_list.c_str(), sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::InputText("##hud", buf, sizeof(buf), ImGuiInputTextFlags_EnterReturnsTrue))
    {
        g_hud_list = buf;
        {
            std::unique_lock lock(g_shaders_lock);
            parse_hud_list();
        }
        save_hud_list();
    }
    ImGui::TextDisabled("HUD pixel shader hashes, comma separated, saved to hudmask.cfg. Press Enter to apply.");

    draw_log();
    draw_finder();
}

//----------------------------------------------------------------------------

static bool register_with_reshade()
{
    if (!reshade::register_addon(g_module))
        return false;
    reshade::register_event<reshade::addon_event::init_effect_runtime>(on_init_effect_runtime);
    reshade::register_event<reshade::addon_event::destroy_effect_runtime>(on_destroy_effect_runtime);
    reshade::register_event<reshade::addon_event::destroy_swapchain>(on_destroy_swapchain);
    reshade::register_event<reshade::addon_event::init_pipeline>(on_init_pipeline);
    reshade::register_event<reshade::addon_event::destroy_pipeline>(on_destroy_pipeline);
    reshade::register_event<reshade::addon_event::bind_pipeline>(on_bind_pipeline);
    reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(on_bind_render_targets);
    reshade::register_event<reshade::addon_event::reset_command_list>(on_reset_command_list);
    reshade::register_event<reshade::addon_event::destroy_command_list>(on_destroy_command_list);
    reshade::register_event<reshade::addon_event::draw>(on_draw);
    reshade::register_event<reshade::addon_event::draw_indexed>(on_draw_indexed);
    reshade::register_event<reshade::addon_event::draw_or_dispatch_indirect>(on_draw_indirect);
    reshade::register_event<reshade::addon_event::reshade_begin_effects>(on_begin_effects);
    reshade::register_event<reshade::addon_event::present>(on_present);
    reshade::register_overlay(nullptr, draw_settings);
    return true;
}

// OptiScaler (and some other setups) load ReShade, unload it, then load it
// again. We stay loaded the whole time, so DllMain never runs a second time and
// we'd never register with the instance that actually renders. Watch the
// loader instead and re-register when ReShade comes back. Games that load it
// once never hit this.
struct ldr_unicode_string { USHORT length, maximum_length; PWSTR buffer; };
struct ldr_dll_notification_data { ULONG flags; const ldr_unicode_string *full_name, *base_name; PVOID base; ULONG size; };
using PFN_LdrDllNotification = VOID(CALLBACK *)(ULONG, const ldr_dll_notification_data *, PVOID);
using PFN_LdrRegisterDllNotification = LONG(NTAPI *)(ULONG, PFN_LdrDllNotification, PVOID, PVOID *);
static const ULONG LDR_DLL_LOADED = 1, LDR_DLL_UNLOADED = 2;
static std::atomic<bool> g_reshade_alive{true};
static std::atomic<HMODULE> g_reshade_module{nullptr};

// Objects made through the old ReShade can't be released through it anymore,
// so just drop them. g_shaders stays: the game's shaders survive the reload and
// the new instance won't report them again.
static void forget_reshade_objects()
{
    g_runtime = nullptr;
    g_fm_ready = false;
    g_fm = frame_mask();
    g_fm_pipeline = {0};
    g_fm_layout = {0};
    g_views.clear();
    g_bound = 0;
}

static DWORD WINAPI reregister_thread(LPVOID)
{
    // Thread start waits for the loader lock, i.e. ReShade's DllMain is done.
    // reshade.hpp caches the first module handle it sees; this relies on the
    // reloaded DLL landing at the same base, which ASLR keeps per boot.
    if (g_reshade_module != reshade::internal::get_reshade_module_handle())
        return 0;
    g_reshade_alive = true;
    if (register_with_reshade())
        reshade::log::message(reshade::log::level::info, "HUD Mask: ReShade was loaded again; registered with the new instance");
    else
        g_reshade_alive = false;
    return 0;
}

static VOID CALLBACK on_dll_notification(ULONG reason, const ldr_dll_notification_data *data, PVOID)
{
    const HMODULE module = static_cast<HMODULE>(data->base);
    if (reason == LDR_DLL_UNLOADED && module == g_reshade_module)
    {
        g_reshade_alive = false;
        forget_reshade_objects();
    }
    else if (reason == LDR_DLL_LOADED && !g_reshade_alive &&
             GetProcAddress(module, "ReShadeRegisterAddon") != nullptr)
    {
        g_reshade_module = module;
        if (HANDLE t = CreateThread(nullptr, 0, reregister_thread, nullptr, 0, nullptr))
            CloseHandle(t);
    }
}

static void watch_reshade_reloads()
{
    g_reshade_module = reshade::internal::get_reshade_module_handle();
    auto reg = reinterpret_cast<PFN_LdrRegisterDllNotification>(
        GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "LdrRegisterDllNotification"));
    static PVOID cookie = nullptr;
    if (reg != nullptr)
        reg(0, on_dll_notification, nullptr, &cookie);
}

BOOL APIENTRY DllMain(HMODULE module, DWORD reason, LPVOID)
{
    switch (reason)
    {
    case DLL_PROCESS_ATTACH:
        g_module = module;
        // Must register before any other ReShade call. reshade.hpp latches the
        // module handle on first use, and only register_addon passes it.
        if (!register_with_reshade())
            return FALSE;
        load_settings();
        watch_reshade_reloads();
        break;
    case DLL_PROCESS_DETACH:
        if (g_reshade_alive)
            reshade::unregister_addon(module);
        break;
    }
    return TRUE;
}
