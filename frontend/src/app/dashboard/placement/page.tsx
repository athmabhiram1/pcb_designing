"use client";

import React, { useState, useMemo, useCallback } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
    Cpu,
    Settings2,
    Target,
    MousePointer2,
    ListFilter,
    ChevronRight,
    ArrowLeft,
    AlertTriangle,
    Layers,
    Loader2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { MechButton } from "@/components/ui/MechButton";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { usePlatform, type GeneratedComponent } from "@/lib/platform-context";
import { api } from "@/lib/api";

// ── Component Type Derivation ─────────────────────────────────────────────
type CompType = "IC" | "R" | "C" | "LED" | "CONN" | "Q" | "L" | "D" | "XTAL" | "MISC";

function deriveType(ref: string): CompType {
    const prefix = ref.replace(/[0-9]/g, "").toUpperCase();
    if (prefix === "U") return "IC";
    if (prefix === "R") return "R";
    if (prefix === "C") return "C";
    if (prefix === "D") return "LED";
    if (prefix === "Q" || prefix === "T") return "Q";
    if (prefix === "J" || prefix === "P") return "CONN";
    if (prefix === "L") return "L";
    if (prefix === "Y") return "XTAL";
    return "MISC";
}

// ── Visual Config per Type (SCALED UP) ────────────────────────────────────
const TYPE_VISUALS: Record<CompType, {
    w: number; h: number; color: string; label: string;
}> = {
    IC: { w: 100, h: 70, color: "#0ea5e9", label: "IC" },
    R: { w: 70, h: 24, color: "#a78bfa", label: "RES" },
    C: { w: 36, h: 50, color: "#34d399", label: "CAP" },
    LED: { w: 44, h: 44, color: "#f97316", label: "LED" },
    CONN: { w: 32, h: 80, color: "#fbbf24", label: "CON" },
    Q: { w: 50, h: 50, color: "#f472b6", label: "FET" },
    L: { w: 56, h: 30, color: "#60a5fa", label: "IND" },
    XTAL: { w: 44, h: 44, color: "#e879f9", label: "OSC" },
    D: { w: 44, h: 44, color: "#f97316", label: "DIO" },
    MISC: { w: 50, h: 50, color: "#6b7280", label: "?" },
};

// ── Node Mapping ──────────────────────────────────────────────────────────
interface PlacedNode {
    id: string;
    x: number;
    y: number;
    type: CompType;
    value: string;
    part: string;
    footprint: string;
}

// ViewBox 1000x700 — generous space
const VB = { W: 1000, H: 700 };
const PAD = 120;

function mapToNodes(components: GeneratedComponent[]): PlacedNode[] {
    const n = components.length;
    if (n === 0) return [];

    const contentW = VB.W - PAD * 2;
    const contentH = VB.H - PAD * 2;

    // Smart grid: prefer wider layouts
    const cols = Math.max(2, Math.ceil(Math.sqrt(n * 1.6)));
    const rows = Math.ceil(n / cols);
    const cellW = contentW / cols;
    const cellH = contentH / Math.max(rows, 1);

    return components.map((c, i) => {
        const col = i % cols;
        const row = Math.floor(i / cols);
        return {
            id: c.ref,
            x: PAD + col * cellW + cellW / 2,
            y: PAD + row * cellH + cellH / 2,
            type: deriveType(c.ref),
            value: c.value,
            part: c.part,
            footprint: c.footprint,
        };
    });
}

// ── SVG Component Shapes ──────────────────────────────────────────────────
function ComponentShape({ node, isSelected, onClick }: { node: PlacedNode; isSelected: boolean; onClick: () => void }) {
    const v = TYPE_VISUALS[node.type];
    const hw = v.w / 2;
    const hh = v.h / 2;
    const sel = isSelected;
    const springCfg = { type: "spring" as const, stiffness: 50, damping: 14 };

    return (
        <g onClick={onClick} className="cursor-pointer">
            {/* Selection glow */}
            {sel && (
                <motion.rect
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 0.5 }}
                    x={node.x - hw - 10} y={node.y - hh - 10}
                    width={v.w + 20} height={v.h + 20}
                    rx={8} fill="none"
                    stroke={v.color} strokeWidth="2" strokeDasharray="6 4"
                />
            )}

            {node.type === "IC" ? (
                <>
                    {/* IC chip body */}
                    <rect
                        x={node.x - hw} y={node.y - hh}
                        width={v.w} height={v.h} rx={4}
                        fill={sel ? `${v.color}25` : `${v.color}10`}
                        stroke={sel ? v.color : `${v.color}35`}
                        strokeWidth={sel ? 2.5 : 1.5}
                    />
                    {/* Notch */}
                    <circle cx={node.x - hw + 14} cy={node.y - hh + 14} r={5}
                        fill={`${v.color}15`} stroke={`${v.color}40`} strokeWidth={1}
                    />
                    {/* Pin stubs — left */}
                    {[0.2, 0.4, 0.6, 0.8].map((f, i) => (
                        <line key={`lp-${i}`}
                            x1={node.x - hw - 12} y1={node.y - hh + v.h * f}
                            x2={node.x - hw} y2={node.y - hh + v.h * f}
                            stroke={`${v.color}25`} strokeWidth={2}
                        />
                    ))}
                    {/* Pin stubs — right */}
                    {[0.2, 0.4, 0.6, 0.8].map((f, i) => (
                        <line key={`rp-${i}`}
                            x1={node.x + hw} y1={node.y - hh + v.h * f}
                            x2={node.x + hw + 12} y2={node.y - hh + v.h * f}
                            stroke={`${v.color}25`} strokeWidth={2}
                        />
                    ))}
                    {/* Internal text */}
                    <text x={node.x} y={node.y + 4} textAnchor="middle"
                        className="font-mono text-[11px] font-bold" fill={`${v.color}60`}
                    >IC</text>
                </>
            ) : node.type === "R" ? (
                <>
                    {/* Resistor body */}
                    <rect
                        x={node.x - hw} y={node.y - hh}
                        width={v.w} height={v.h} rx={3}
                        fill={sel ? `${v.color}25` : `${v.color}08`}
                        stroke={sel ? v.color : `${v.color}35`}
                        strokeWidth={sel ? 2.5 : 1.5}
                    />
                    {/* Color bands */}
                    {[0.2, 0.38, 0.56, 0.78].map((f, i) => (
                        <line key={`band-${i}`}
                            x1={node.x - hw + v.w * f} y1={node.y - hh + 3}
                            x2={node.x - hw + v.w * f} y2={node.y + hh - 3}
                            stroke={i === 3 ? "#c4b5fd" : `${v.color}${i === 0 ? "70" : "45"}`}
                            strokeWidth={3}
                        />
                    ))}
                    {/* Leads */}
                    <line x1={node.x - hw - 16} y1={node.y} x2={node.x - hw} y2={node.y}
                        stroke={`${v.color}25`} strokeWidth={2.5} />
                    <line x1={node.x + hw} y1={node.y} x2={node.x + hw + 16} y2={node.y}
                        stroke={`${v.color}25`} strokeWidth={2.5} />
                </>
            ) : node.type === "C" ? (
                <>
                    {/* Capacitor — two parallel plates */}
                    <line x1={node.x} y1={node.y - hh - 10} x2={node.x} y2={node.y - 7}
                        stroke={`${v.color}25`} strokeWidth={2.5} />
                    <line x1={node.x - 16} y1={node.y - 7} x2={node.x + 16} y2={node.y - 7}
                        stroke={sel ? v.color : `${v.color}70`} strokeWidth={4} />
                    <line x1={node.x - 16} y1={node.y + 7} x2={node.x + 16} y2={node.y + 7}
                        stroke={sel ? v.color : `${v.color}70`} strokeWidth={4} />
                    <line x1={node.x} y1={node.y + 7} x2={node.x} y2={node.y + hh + 10}
                        stroke={`${v.color}25`} strokeWidth={2.5} />
                    {/* Polarity + */}
                    <text x={node.x + 20} y={node.y - 12} className="font-mono text-[12px] font-bold" fill={`${v.color}40`}>+</text>
                </>
            ) : node.type === "LED" || node.type === "D" ? (
                <>
                    {/* Diode/LED triangle */}
                    <polygon
                        points={`${node.x},${node.y - 16} ${node.x - 16},${node.y + 12} ${node.x + 16},${node.y + 12}`}
                        fill={sel ? `${v.color}30` : `${v.color}12`}
                        stroke={sel ? v.color : `${v.color}45`}
                        strokeWidth={sel ? 2.5 : 1.5}
                    />
                    <line x1={node.x - 16} y1={node.y + 12} x2={node.x + 16} y2={node.y + 12}
                        stroke={sel ? v.color : `${v.color}60`} strokeWidth={3} />
                    {/* Leads */}
                    <line x1={node.x} y1={node.y - 16} x2={node.x} y2={node.y - 28}
                        stroke={`${v.color}25`} strokeWidth={2.5} />
                    <line x1={node.x} y1={node.y + 12} x2={node.x} y2={node.y + 28}
                        stroke={`${v.color}25`} strokeWidth={2.5} />
                    {/* LED glow arrows */}
                    {node.type === "LED" && (
                        <>
                            <line x1={node.x + 18} y1={node.y - 8} x2={node.x + 28} y2={node.y - 18}
                                stroke={v.color} strokeWidth={1.5} opacity={0.6} />
                            <polygon points={`${node.x + 28},${node.y - 18} ${node.x + 24},${node.y - 14} ${node.x + 26},${node.y - 20}`}
                                fill={v.color} opacity={0.5} />
                            <line x1={node.x + 22} y1={node.y - 3} x2={node.x + 32} y2={node.y - 13}
                                stroke={v.color} strokeWidth={1.5} opacity={0.4} />
                        </>
                    )}
                </>
            ) : node.type === "Q" ? (
                <>
                    {/* Transistor circle */}
                    <circle cx={node.x} cy={node.y} r={22}
                        fill={sel ? `${v.color}20` : `${v.color}06`}
                        stroke={sel ? v.color : `${v.color}35`}
                        strokeWidth={sel ? 2.5 : 1.5}
                    />
                    {/* Internal vertical bar */}
                    <line x1={node.x - 6} y1={node.y - 12} x2={node.x - 6} y2={node.y + 12}
                        stroke={`${v.color}50`} strokeWidth={2.5} />
                    {/* 3 pins: B, C, E */}
                    <line x1={node.x - 30} y1={node.y} x2={node.x - 6} y2={node.y}
                        stroke={`${v.color}30`} strokeWidth={2.5} />
                    <line x1={node.x - 6} y1={node.y - 8} x2={node.x + 14} y2={node.y - 20}
                        stroke={`${v.color}30`} strokeWidth={2} />
                    <line x1={node.x - 6} y1={node.y + 8} x2={node.x + 14} y2={node.y + 20}
                        stroke={`${v.color}30`} strokeWidth={2} />
                    {/* Arrow on emitter */}
                    <polygon points={`${node.x + 14},${node.y + 20} ${node.x + 6},${node.y + 14} ${node.x + 10},${node.y + 22}`}
                        fill={`${v.color}40`} />
                </>
            ) : node.type === "CONN" ? (
                <>
                    {/* Connector */}
                    <rect x={node.x - hw} y={node.y - hh}
                        width={v.w} height={v.h} rx={3}
                        fill={sel ? `${v.color}20` : `${v.color}06`}
                        stroke={sel ? v.color : `${v.color}30`}
                        strokeWidth={sel ? 2.5 : 1.5}
                    />
                    {[0.15, 0.35, 0.55, 0.75, 0.95].map((f, i) => (
                        <circle key={`pin-${i}`} cx={node.x} cy={node.y - hh + v.h * f} r={3.5}
                            fill={`${v.color}40`} stroke={`${v.color}60`} strokeWidth={1}
                        />
                    ))}
                </>
            ) : (
                <>
                    {/* Generic */}
                    <rect x={node.x - hw} y={node.y - hh}
                        width={v.w} height={v.h} rx={8}
                        fill={sel ? `${v.color}20` : `${v.color}06`}
                        stroke={sel ? v.color : `${v.color}25`}
                        strokeWidth={sel ? 2.5 : 1.5}
                    />
                </>
            )}

            {/* Reference label */}
            <text x={node.x} y={node.y + hh + 24} textAnchor="middle"
                className="font-mono text-[13px] font-bold"
                fill={sel ? v.color : "rgba(255,255,255,0.4)"}
            >
                {node.id}
            </text>

            {/* Value label */}
            <text x={node.x} y={node.y + hh + 40} textAnchor="middle"
                className="font-mono text-[10px]"
                fill="rgba(255,255,255,0.18)"
            >
                {node.value}
            </text>
        </g>
    );
}

// ── Type Badge ────────────────────────────────────────────────────────────
const TypeBadge = ({ type }: { type: CompType }) => {
    const v = TYPE_VISUALS[type];
    return (
        <Badge variant="outline"
            className="text-[8px] px-1.5 h-4 uppercase font-bold"
            style={{ borderColor: `${v.color}40`, color: v.color, backgroundColor: `${v.color}10` }}
        >
            {v.label}
        </Badge>
    );
};

// ── Main Page ─────────────────────────────────────────────────────────────
export default function PlacementPage() {
    const router = useRouter();
    const { activeComponents, setActiveComponents, addLog, generationResult } = usePlatform();
    const [selectedId, setSelectedId] = useState<string | null>(null);
    const [isOptimizing, setIsOptimizing] = useState(false);
    const [nodes, setNodes] = useState<PlacedNode[]>([]);
    const [hasInit, setHasInit] = useState(false);

    React.useEffect(() => {
        if (activeComponents.length > 0 && !hasInit) {
            setNodes(mapToNodes(activeComponents));
            setSelectedId(activeComponents[0]?.ref || null);
            setHasInit(true);
        }
    }, [activeComponents, hasInit]);

    const selectedNode = useMemo(() => nodes.find(c => c.id === selectedId), [selectedId, nodes]);

    const handleOptimize = useCallback(async () => {
        if (isOptimizing || nodes.length === 0) return;
        setIsOptimizing(true);
        const t = () => new Date().toLocaleTimeString("en-US", { hour12: false });
        addLog({ time: t(), type: "AI", msg: "Launching RL placement optimizer…" });

        try {
            const result = await api.optimizePlacement({
                components: activeComponents.map(c => ({
                    ref: c.ref, value: c.value,
                    x: c.x, y: c.y, rotation: c.rotation, layer: "top",
                })),
                board_width: 100, board_height: 80,
            });

            if (result.success && result.positions) {
                addLog({ time: t(), type: "AI", msg: `✓ Optimized: ${result.improvement || "Grid layout applied"}` });
                const contentW = VB.W - PAD * 2;
                const contentH = VB.H - PAD * 2;
                setNodes(prev => prev.map(n => {
                    const pos = result.positions[n.id];
                    if (!pos) return n;
                    return {
                        ...n,
                        x: (pos.x / 100) * contentW + PAD,
                        y: (pos.y / 80) * contentH + PAD,
                    };
                }));
                setActiveComponents(activeComponents.map(c => {
                    const pos = result.positions[c.ref];
                    return pos ? { ...c, x: pos.x, y: pos.y, rotation: pos.rotation } : c;
                }));
            } else {
                addLog({ time: t(), type: "ERROR", msg: result.error || "Optimization failed" });
            }
        } catch (err: any) {
            addLog({ time: t(), type: "ERROR", msg: `Network: ${err.message}` });
        } finally {
            setIsOptimizing(false);
        }
    }, [isOptimizing, nodes, activeComponents, addLog, setActiveComponents]);

    // ── Empty State ──
    if (activeComponents.length === 0) {
        return (
            <div className="h-full flex flex-col items-center justify-center gap-6 relative font-sans">
                <div className="absolute inset-0 bento-grid opacity-5 pointer-events-none" />
                <div className="relative z-10 text-center space-y-4">
                    <div className="w-20 h-20 rounded-2xl bg-primary/5 border border-primary/15 flex items-center justify-center mx-auto">
                        <Layers className="w-10 h-10 text-primary/30" />
                    </div>
                    <h2 className="text-xl font-display font-black text-white uppercase tracking-tight">
                        No Components <span className="text-primary">Loaded</span>
                    </h2>
                    <p className="text-xs text-white/30 font-mono max-w-xs mx-auto">Generate a circuit first to populate the placement canvas.</p>
                    <MechButton variant="primary" onClick={() => router.push("/dashboard")}>
                        <ArrowLeft className="w-4 h-4" /> Back to Mission Control
                    </MechButton>
                </div>
            </div>
        );
    }

    return (
        <div className="h-full flex flex-col overflow-hidden relative font-sans">
            {/* Header */}
            <header className="h-14 border-b border-border-subtle flex items-center justify-between px-6 bg-dark-800/40 backdrop-blur-xl z-20 shrink-0">
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2 text-primary font-mono text-[9px] uppercase tracking-[0.3em]">
                        <Target className="w-3 h-3" /> Spatial Engine
                    </div>
                    <div className="h-4 w-px bg-white/10" />
                    <h1 className="text-sm font-display font-black text-white uppercase tracking-tight">
                        Platform <span className="text-primary">Placement</span>
                    </h1>
                </div>

                <div className="flex items-center gap-4">
                    <span className="text-[10px] font-mono text-white/30">{nodes.length} components</span>
                    {generationResult?.description && (
                        <span className="text-[10px] font-mono text-primary/60 truncate max-w-[200px]">{generationResult.description}</span>
                    )}
                    <div className="h-5 w-px bg-white/10" />
                    <MechButton variant="primary" size="sm" onClick={handleOptimize} disabled={isOptimizing}>
                        {isOptimizing ? <><Loader2 className="w-3 h-3 animate-spin" /> Optimizing…</> : "Auto-Optimize"}
                    </MechButton>
                    <button className="p-2 bg-dark-800 border border-white/5 rounded-lg hover:border-primary/30 transition-all text-white/40 hover:text-primary">
                        <Settings2 className="w-4 h-4" />
                    </button>
                </div>
            </header>

            {/* Main */}
            <main className="flex-1 flex overflow-hidden relative">

                {/* Left: Component List */}
                <div className="w-56 border-r border-white/5 flex flex-col bg-dark-800/20 shrink-0">
                    <div className="p-4 border-b border-white/5 flex items-center justify-between">
                        <span className="text-[9px] font-mono text-white/30 uppercase tracking-[0.2em] flex items-center gap-2">
                            <ListFilter className="w-3 h-3 text-primary" /> Registry
                        </span>
                        <Badge variant="secondary" className="bg-dark-900 border-white/10 text-[9px]">{nodes.length}</Badge>
                    </div>
                    <ScrollArea className="flex-1">
                        <div className="p-2 space-y-0.5">
                            {nodes.map(n => {
                                const v = TYPE_VISUALS[n.type];
                                return (
                                    <button key={n.id} onClick={() => setSelectedId(n.id)}
                                        className={cn(
                                            "w-full flex items-center justify-between p-2.5 rounded-lg border transition-all text-left",
                                            selectedId === n.id
                                                ? "bg-white/5 border-white/10 text-white"
                                                : "bg-transparent border-transparent text-white/35 hover:bg-white/[0.03] hover:text-white/50"
                                        )}
                                    >
                                        <div className="flex items-center gap-2">
                                            <div className="w-6 h-6 rounded flex items-center justify-center border"
                                                style={{
                                                    borderColor: selectedId === n.id ? `${v.color}60` : `${v.color}20`,
                                                    backgroundColor: selectedId === n.id ? `${v.color}15` : "transparent",
                                                }}
                                            >
                                                <Cpu className="w-3 h-3" style={{ color: selectedId === n.id ? v.color : `${v.color}40` }} />
                                            </div>
                                            <div>
                                                <div className="text-[11px] font-mono font-bold leading-none mb-0.5">{n.id}</div>
                                                <div className="text-[9px] font-mono text-white/20">{n.value}</div>
                                            </div>
                                        </div>
                                        <TypeBadge type={n.type} />
                                    </button>
                                );
                            })}
                        </div>
                    </ScrollArea>
                </div>

                {/* Center: Canvas — FILLS all remaining space */}
                <div className="flex-1 relative bg-dark-900/50 overflow-hidden">
                    {/* Title bar */}
                    <div className="absolute top-0 inset-x-0 h-8 bg-dark-800/50 border-b border-white/5 flex items-center justify-between px-4 z-10">
                        <span className="text-[9px] font-mono text-white/20 uppercase tracking-[0.2em]">PCB Layout Canvas</span>
                        <div className="flex gap-3 text-[9px] font-mono text-white/15">
                            <span>100 × 80 mm</span>
                        </div>
                    </div>

                    {/* SVG fills the entire panel */}
                    <svg
                        viewBox={`0 0 ${VB.W} ${VB.H}`}
                        preserveAspectRatio="xMidYMid meet"
                        className="absolute inset-0 w-full h-full pt-8"
                    >
                        <defs>
                            <pattern id="pcbGrid" width="50" height="50" patternUnits="userSpaceOnUse">
                                <path d="M 50 0 L 0 0 0 50" fill="none" stroke="rgba(14,165,233,0.06)" strokeWidth="0.5" />
                            </pattern>
                        </defs>

                        <rect width={VB.W} height={VB.H} fill="url(#pcbGrid)" />

                        {/* Board outline */}
                        <rect x={PAD - 20} y={PAD - 20}
                            width={VB.W - PAD * 2 + 40} height={VB.H - PAD * 2 + 40}
                            rx={10} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="1.5" strokeDasharray="8 5"
                        />

                        {/* Ratsnest lines — connect every component to the first IC or first node */}
                        {(() => {
                            const hub = nodes.find(n => n.type === "IC") || nodes[0];
                            if (!hub || nodes.length < 2) return null;
                            return nodes.filter(n => n.id !== hub.id).map((n, i) => (
                                <line key={`rat-${i}`}
                                    x1={hub.x} y1={hub.y} x2={n.x} y2={n.y}
                                    stroke="rgba(14,165,233,0.06)" strokeWidth="1" strokeDasharray="4 6"
                                />
                            ));
                        })()}

                        {/* Components */}
                        {nodes.map(n => (
                            <ComponentShape key={n.id} node={n} isSelected={selectedId === n.id} onClick={() => setSelectedId(n.id)} />
                        ))}
                    </svg>
                </div>

                {/* Right: Properties Panel */}
                <AnimatePresence>
                    {selectedNode && (
                        <motion.div
                            initial={{ width: 0, opacity: 0 }}
                            animate={{ width: 260, opacity: 1 }}
                            exit={{ width: 0, opacity: 0 }}
                            transition={{ type: "spring", stiffness: 200, damping: 25 }}
                            className="border-l border-white/5 bg-dark-800/30 overflow-hidden shrink-0"
                        >
                            <div className="w-[260px] p-5 space-y-5 h-full flex flex-col">
                                <div>
                                    <div className="flex items-center justify-between mb-2">
                                        <h2 className="text-lg font-display font-black text-white uppercase">{selectedNode.id}</h2>
                                        <MousePointer2 className="w-4 h-4 text-primary/50" />
                                    </div>
                                    <p className="text-[11px] text-white/40 mb-2">{selectedNode.part} — {selectedNode.value}</p>
                                    <div className="flex gap-2">
                                        <TypeBadge type={selectedNode.type} />
                                        <Badge className="bg-white/5 border-white/10 text-white/40 text-[9px]">TOP</Badge>
                                    </div>
                                </div>

                                <div className="grid grid-cols-2 gap-2.5">
                                    {[
                                        { l: "Pos X", v: `${Math.round(selectedNode.x)}` },
                                        { l: "Pos Y", v: `${Math.round(selectedNode.y)}` },
                                        { l: "Part", v: selectedNode.part || "—" },
                                        { l: "Type", v: TYPE_VISUALS[selectedNode.type].label },
                                    ].map(s => (
                                        <div key={s.l} className="p-2 bg-dark-900 rounded-lg border border-white/5">
                                            <div className="text-[8px] text-white/15 uppercase mb-0.5 tracking-widest">{s.l}</div>
                                            <div className="text-[11px] font-mono text-white/70">{s.v}</div>
                                        </div>
                                    ))}
                                </div>

                                {selectedNode.footprint && (
                                    <div className="p-2.5 bg-dark-900 rounded-lg border border-white/5">
                                        <div className="text-[8px] text-white/15 uppercase mb-0.5 tracking-widest">Footprint</div>
                                        <div className="text-[10px] font-mono text-white/50 break-all">{selectedNode.footprint}</div>
                                    </div>
                                )}

                                <div className="mt-auto">
                                    <div className="p-3 rounded-lg bg-orange-500/5 border border-orange-500/10 flex items-start gap-2 mb-3">
                                        <AlertTriangle className="w-3.5 h-3.5 text-orange-400 mt-0.5 shrink-0" />
                                        <div>
                                            <div className="text-[9px] font-bold text-orange-400 uppercase">DFM Note</div>
                                            <p className="text-[9px] text-orange-400/50 leading-relaxed">Check clearance after optimization.</p>
                                        </div>
                                    </div>
                                    <MechButton variant="primary" className="w-full text-xs font-bold group">
                                        Re-Anchor <ChevronRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-all" />
                                    </MechButton>
                                </div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </main>
        </div>
    );
}
