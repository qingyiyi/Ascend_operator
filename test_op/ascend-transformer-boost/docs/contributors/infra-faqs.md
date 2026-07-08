
#### 1.  **请问提交PR后出现cann-cla/no红色标签，如何处理？**

出现该标签表示该PR中所包含的commit中，有部分贡献者没有签署CANN社区的贡献者协议CLA，签署地址可以在PR评论区找到；

1）如果是以个人名义贡献者，请选择“签署个人CLA”；

2）如果是以一家企业名义贡献，请选择“签署法人CLA”；

3）如果是以一家已签署企业CLA的企业员工名义贡献，请选择“法人贡献者登记”；

签署后会收到主题是Signing CLA on project of xx的邮件，请联系邮件内容里的Corporation Managers进行审批，审批通过后，在PR评论区评论/check-cla，重新触发CLA检查就会打上cann-cla/yes标签。CLA检查是使用commit信息中的committer邮箱作为检查凭证的。该邮箱可以通过git log --pretty=fuller查询到。
		
<table>
<tbody><tr>
<th>场景</th>
<th>选择</th>
<th>处理方案</th>
</tr>
<tr>
<td>commit邮箱和Gitcode提交邮箱一致</td>
<td>统一用该邮箱</td>
<td>使用该邮箱直接在上述“签署地址”签署CLA即可</td>
</tr>
<tr>
<td rowspan="2">commit邮箱和Gitcode提交邮箱不一致</td>
<td>希望使用commit邮箱签署</td>
<td>调整Gitcode提交邮箱为commit邮箱，在Gitcode个人设置页面添加commit邮箱并设置为提交邮箱，然后在上述签署地址完成CLA签署即可</td>
</tr>
<tr>
<td>希望使用Gitcode提交邮箱签署</td>
<td>在git运行的本地通过 git config --global user.name **** 和 git config --global user.email ****修改配置可调整git的commit邮箱信息为Gitcode的提交邮箱，完成后再进入签署地址进行CLA签署</td>
</tr>
</tbody>
</table>

#### 2.  **请问为什么不能fork一个CANN/abc仓库到个人账号下？**

这个问题通常是因为在您的个人账号下，已经有abc的同名仓库，比如你之前已经从CANN组织下fork了名称为abc的仓库；因为Gitcode是通过你的个人账号加仓库名寻址的，所以不允许在你的个人账号下有同名仓库。

解决方法：修改个人账户下已有仓库的名称和路径，然后再从“CANN/abc”仓库fork即可。

#### 3.  **请问保护分支和非保护分支的区别？**

保护分支可以设置特定的角色或者成员具有该分支的推送权限和合并权限；非保护分支则不支持。

#### 4.  **请问社区的CANN开发者角色可以直接push代码到仓库吗？**

CANN开发者不允许直接push代码到社区仓库，只有仓库管理员可以push代码到社区仓库；
如果CANN开发者想要贡献代码，只能fork社区仓库到个人账号下，通过提交Pull Request方式贡献代码。

#### 5.  **请问直接push代码到仓库和通过评论/lgtm 、/approve合入代码有何区别？**

通过git命令直接push代码到仓库缺少必要的审核环节，存在一定合入风险；主要应用场景是比如需要上传的文件过大超过个人仓库限制，只能通过直接push到仓库的非保护分支，然后再通过非保护分支往保护分支merge；

通过评论/lgtm /approve合入代码从流程上增加了评审环节，保证一份代码的合入至少需要提交者以外的一位committer的评审同意，即便提交者本人是committer也需要另一位committer同意。

#### 6.  **请问CANN社区仓库评论区都支持哪些命令，分别都是什么含义？**

目前社区仓库评论区主要支持的命令请参见[CANN社区评论命令一览](infra-command.md)。

#### 7.  **请问我提交PR后为什么没有触发CI构建，需要如何处理？**

CI未及时触发通常有两种情况：

- 第一种可能是网络原因或系统任务调度原因，导致从代码仓库发出的webhook通知事件没有及时到达目标服务，所以没有触发CI构建；这种情况可以通过在PR评论区评论 **/compile** 重新触发。
	
- 第二种可能是代码仓库创建以后短时间内提交PR，此时jenkins服务器侧尚未创建CI构建工程，所以触发不到CI构建，评论 **/compile** 也不生效，这种情况请稍等一下系统自动构建工程。
