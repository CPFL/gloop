/*
  Copyright (C) 2015 Yusuke Suzuki <yusuke.suzuki@sslab.ics.keio.ac.jp>

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
  THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Regex.h"
#include <iostream>

// http://eli.thegreenplace.net/2012/06/08/basic-source-to-source-transformation-with-clang/
// http://eli.thegreenplace.net/2014/05/01/modern-source-to-source-transformation-with-clang-and-libtooling
// http://clang.llvm.org/doxygen/classclang_1_1RecursiveASTVisitor.html
// http://clang.llvm.org/docs/IntroductionToTheClangAST.html
// http://clang.llvm.org/docs/RAVFrontendAction.html
class LambdaVisitor : public clang::RecursiveASTVisitor<LambdaVisitor> {
public:
    LambdaVisitor(clang::Rewriter& rewriter)
        : m_rewriter(rewriter)
    {
    }

    bool VisitStmt(clang::Stmt* stmt)
    {
        if (clang::LambdaExpr* lambdaExpr = llvm::dyn_cast<clang::LambdaExpr>(stmt)) {
            // Found lambda expression.
        }
        return true;
    }

private:
    clang::Rewriter& m_rewriter;
};

class LambdaRewriter : public clang::ASTConsumer {
public:
    LambdaRewriter(clang::CompilerInstance& CI)
        : m_CI(CI)
        , m_rewriter()
        , m_visitor(m_rewriter)
    {
    }

    virtual void Initialize(clang::ASTContext &context)
    {
        auto& sourceManager = context.getSourceManager();
        m_rewriter.setSourceMgr(sourceManager, m_CI.getLangOpts());
        m_mainFileID = sourceManager.getMainFileID();
    }

    virtual bool HandleTopLevelDecl(clang::DeclGroupRef DG) override
    {
        for (auto& decl : DG) {
            m_visitor.TraverseDecl(decl);
        }
        return true;
    }

    virtual void HandleTranslationUnit(clang::ASTContext& context) override
    {
        auto& sourceManager = context.getSourceManager();
        // http://clang.llvm.org/doxygen/classclang_1_1Rewriter.html#a6345f29dfc642152bf0d51eae32f900e
        if (const clang::RewriteBuffer* rewriteBuffer = m_rewriter.getRewriteBufferFor(m_mainFileID)) {
            std::cout << std::string(rewriteBuffer->begin(), rewriteBuffer->end());
        } else {
            // No modification is made.
            llvm::StringRef buffer = sourceManager.getBufferData(m_mainFileID);
            std::cout << std::string(buffer.begin(), buffer.end());
        }
    }

private:
    clang::CompilerInstance& m_CI;
    clang::Rewriter m_rewriter;
    clang::FileID m_mainFileID;
    LambdaVisitor m_visitor;
};

class RewriteLambdaAction : public clang::PluginASTAction {
protected:
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef InFile) override
    {
        return llvm::make_unique<LambdaRewriter>(CI);
    }

    bool ParseArgs(const clang::CompilerInstance &CI, const std::vector<std::string> &args) override
    {
        for (const auto& arg : args) {
            llvm::errs() << "PrintFunctionNames arg = " << arg << "\n";

            // Example error handling.
            clang::DiagnosticsEngine &D = CI.getDiagnostics();
            if (arg == "-an-error") {
                unsigned DiagID = D.getCustomDiagID(clang::DiagnosticsEngine::Error, "invalid argument '%0'");
                D.Report(DiagID) << arg;
                return false;
            }
        }

        if (!args.empty() && args[0] == "help") {
            PrintHelp(llvm::errs());
        }

        return true;
    }

    void PrintHelp(llvm::raw_ostream &ros)
    {
    }
};

static clang::FrontendPluginRegistry::Add<RewriteLambdaAction> X("rewrite-lambda", "translate lambda for portable one before passing the source to NVCC");
