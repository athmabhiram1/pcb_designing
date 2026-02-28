"use client";

import { Sidebar } from "@/components/layout/Sidebar";
import { NodeCanvas } from "@/components/ui/NodeCanvas";
import { PlatformProvider } from "@/lib/platform-context";
import { motion, AnimatePresence } from "framer-motion";

export default function DashboardLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <PlatformProvider>
            <div className="flex h-screen w-full bg-dark-900 text-foreground overflow-hidden font-sans">
                <NodeCanvas />

                {/* Navigation Sidebar */}
                <Sidebar />

                {/* Main Content */}
                <main className="flex-1 flex flex-col min-w-0 relative z-10">

                    {/* Top Header Bar */}
                    <header className="h-14 flex items-center justify-between px-6 glass-nav z-20 flex-shrink-0">
                        <div className="flex items-center gap-3">
                            <div className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse shadow-[0_0_6px_#0ea5e9]" />
                            <span className="text-[10px] font-mono text-white/50 uppercase tracking-[0.25em]">
                                System Operational
                            </span>
                        </div>

                        <div className="flex items-center gap-5">
                            <div className="hidden md:flex flex-col items-end">
                                <span className="text-[8px] text-white/30 uppercase tracking-[0.3em] font-mono">Current Project</span>
                                <span className="text-sm font-display text-white font-semibold tracking-tight leading-none mt-0.5">
                                    Aether_Flight_Controller_v4
                                </span>
                            </div>
                            {/* Avatar */}
                            <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-primary/40 to-secondary/40 border border-primary/25 flex items-center justify-center">
                                <div className="w-4 h-4 rounded-full bg-gradient-to-tr from-primary to-secondary opacity-70" />
                            </div>
                        </div>
                    </header>

                    {/* Scrollable Content */}
                    <div className="flex-1 overflow-auto relative">
                        <AnimatePresence mode="wait">
                            <motion.div
                                key="dashboard-content"
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -8 }}
                                transition={{ duration: 0.25, ease: "easeOut" }}
                                className="h-full"
                            >
                                {children}
                            </motion.div>
                        </AnimatePresence>
                    </div>

                    {/* Footer Status Bar */}
                    <footer className="h-7 bg-dark-800/80 border-t border-border-subtle flex items-center justify-between px-5 text-[9px] font-mono text-white/25 z-20 flex-shrink-0">
                        <div className="flex items-center gap-5">
                            <span className="flex items-center gap-1.5">
                                <div className="w-1 h-1 rounded-full bg-emerald-500/60" />
                                CLOUD_SYNC: ONLINE
                            </span>
                            <span>LATENCY: 14ms</span>
                        </div>
                        <div className="flex items-center gap-5">
                            <span className="hidden sm:inline">TS_2024.02.24_02:33:00</span>
                            <span className="text-primary/50">AUTOSAVE_ACTIVE</span>
                        </div>
                    </footer>
                </main>
            </div>
        </PlatformProvider>
    );
}
